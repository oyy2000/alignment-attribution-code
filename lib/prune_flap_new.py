import torch
import torch.nn as nn
from tqdm import tqdm
from .data import get_loaders
from .prune import prepare_calibration_input as wanda_prepare_calib
from .prune import find_layers


class FlapStatWrapper:
    """Collect input activation statistics (mean, var) per feature for a Linear layer.
    Works similarly to BiasGPT in the earlier prototype but lighter and compatible with wanda-style loop."""

    def __init__(self, layer):
        self.layer = layer
        self.inp_tokens = []  # list of (tokens, hidden)

    def add_batch(self, inp, out, tar=None):  # tar unused but keep signature parity
        # inp arrives as (batch, seq, hidden); flatten batch*seq
        if inp.dim() == 3:
            flat = inp.reshape(-1, inp.size(-1))
        elif inp.dim() == 2:  # (seq, hidden)
            flat = inp
        else:  # (hidden)
            flat = inp.unsqueeze(0)
        self.inp_tokens.append(flat.detach().to(torch.float32))

    @property
    def var(self):
        if not self.inp_tokens:
            return torch.zeros(self.layer.weight.shape[1], device=self.layer.weight.device)
        X = torch.cat(self.inp_tokens, dim=0)  # (N, hidden)
        return torch.var(X, dim=0, unbiased=False) + 1e-8

    @property
    def mean(self):
        if not self.inp_tokens:
            return torch.zeros(self.layer.weight.shape[1], device=self.layer.weight.device)
        X = torch.cat(self.inp_tokens, dim=0)
        return torch.mean(X, dim=0)


def _compute_flap_metric(metric_name, wrapper: FlapStatWrapper, layer: nn.Linear):
    # layer.weight: (out_features, in_features)
    if metric_name == 'IFV':
        return wrapper.var  # (in_features,)
    if metric_name == 'WIFV':
        # Sum of squared weights per input feature (column-wise) times variance
        col_pow2 = torch.sum(layer.weight.data.pow(2), dim=0)
        return wrapper.var * col_pow2
    if metric_name == 'WIFN':
        # Mean over output neurons of |w| * sqrt(var)
        return (torch.abs(layer.weight.data) * torch.sqrt(wrapper.var.reshape(1, -1))).mean(dim=0)
    raise ValueError(f"Unknown FLAP metric {metric_name}")


def prune_flap(
    args,
    model,
    tokenizer,
    model_base=None,  # unused; kept for signature similarity
    device=torch.device("cuda:0"),
    prune_n=0,
    prune_m=0,
    prune_data=None,
):
    """Wanda-style FLAP pruning: collect per-layer input stats then prune heads (q,k,v family via o_proj proxy) and MLP neurons.
    Currently only masks (zeros) selected rows consistent with sparsity_ratio per layer.
    """
    if prune_data is None:
        prune_data = getattr(args, 'prune_data', 'wikitext')
    if not hasattr(args, 'metrics'):
        args.metrics = 'IFV'
    metric_name = args.metrics
    if not hasattr(args, 'disentangle'):
        args.disentangle = True

    use_cache = model.config.use_cache
    model.config.use_cache = False
    print(f"loading calibration data {prune_data} (FLAP)")
    dataloader, _ = get_loaders(
        prune_data,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
        disentangle=args.disentangle,
    )
    print("dataset loading complete (FLAP)")

    with torch.no_grad():
        inps, outs, tars, attention_mask, position_ids = wanda_prepare_calib(
            model, dataloader, device, args.nsamples
        )
    if not args.disentangle:
        tars = [torch.zeros_like(tar) for tar in tars]

    inps = [inp.squeeze(0).to(device) for inp in inps]
    tars = [tar.squeeze(0).to(device) for tar in tars]
    attention_mask = [am.to(device) for am in attention_mask]
    position_ids = [pids.to(device) for pids in position_ids]

    layers = model.model.layers
    print("prune every linear layer (FLAP style)")

    # We'll gather per-layer metrics for self_attn.o_proj (head proxy) and mlp.down_proj (neuron proxy)
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        # focus only on target projection layers; skip others
        target_names = []
        if 'self_attn.o_proj' in subset:
            target_names.append('self_attn.o_proj')
        if 'mlp.down_proj' in subset:
            target_names.append('mlp.down_proj')
        if not target_names:
            continue

        wrapped = {name: FlapStatWrapper(subset[name]) for name in target_names}

        def add_batch(name, tar):
            def hook(_, inp, out):
                wrapped[name].add_batch(inp[0].data, out.data, tar)
            return hook

        for j in range(args.nsamples):
            handles = []
            for name in wrapped:
                handles.append(subset[name].register_forward_hook(add_batch(name, tars[j])))
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0]
            for h in handles:
                h.remove()

        # Compute metrics (and optionally prune)
        for name in target_names:
            lin = subset[name]
            metric = _compute_flap_metric(metric_name, wrapped[name], lin)
            # Dump score logic
            if getattr(args, 'dump_flap_score', False):
                import os, pickle
                # folder naming similar style to wanda
                variant_tag = metric_name.lower()
                if getattr(args, 'disentangle', False):
                    subdir = f"flap_score/{prune_data}_{variant_tag}_disentangle"
                else:
                    subdir = f"flap_score/{prune_data}_{variant_tag}"
                save_folder = os.path.join(args.save, subdir)
                if not os.path.exists(save_folder):
                    os.makedirs(save_folder, exist_ok=True)
                target_file = os.path.join(
                    save_folder,
                    f"flap_metric_layer_{i}_name_{name}_{prune_data}_{variant_tag}.pkl",
                )
                with open(target_file, 'wb') as f:
                    pickle.dump(metric.cpu(), f)
                # also torch.save
                torch.save(metric.cpu(), target_file.replace('.pkl', '_torch.pt'))
                print(f"[FLAP] Saved metric layer {i} {name} to {target_file}")
                # Skip pruning if only dumping
                continue
            # For o_proj treat contiguous head blocks of size 128 like original FLAP; for mlp we prune individual neurons
            if name == 'self_attn.o_proj':
                # metric over input features; convert to head scores by reshaping
                try:
                    metric_heads = metric.reshape(-1, 128).mean(dim=1)
                except Exception:
                    metric_heads = metric  # fallback
                k_keep = max(1, int(metric_heads.numel() * (1 - args.sparsity_ratio)))
                keep_idx = torch.topk(metric_heads, k_keep).indices
                mask_heads = torch.zeros_like(metric_heads, dtype=torch.bool)
                mask_heads[keep_idx] = True
                expand_mask = mask_heads.repeat_interleave(128)
                # zero pruned columns (input features) in q,k,v handled indirectly by zeroing o_proj rows? Here zero rows of o_proj weight not columns.
                # We'll zero rows (output neurons) mapped by head blocks for simplicity.
                # Map expand_mask length to out_features if possible
                if lin.weight.data.shape[0] == expand_mask.numel():
                    row_mask = ~expand_mask.to(lin.weight.device)
                    lin.weight.data[row_mask] = 0
                else:
                    # fallback: prune smallest individual input features
                    flat_metric = metric
                    k_feat = max(1, int(flat_metric.numel() * (1 - args.sparsity_ratio)))
                    top_feat = torch.topk(flat_metric, k_feat).indices
                    feat_mask = torch.ones_like(flat_metric, dtype=torch.bool)
                    feat_mask[top_feat] = False
                    lin.weight.data[:, feat_mask] = 0
            elif name == 'mlp.down_proj':
                neuron_metric = metric  # size in_features
                k_keep = max(1, int(neuron_metric.numel() * (1 - args.sparsity_ratio)))
                keep_idx = torch.topk(neuron_metric, k_keep).indices
                prune_mask = torch.ones_like(neuron_metric, dtype=torch.bool)
                prune_mask[keep_idx] = False
                lin.weight.data[:, prune_mask] = 0

        # swap buffers like wanda to propagate
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_mask[j],
                    position_ids=position_ids[j],
                )[0].squeeze(0)
        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
