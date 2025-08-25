import torch
import torch.nn as nn
from .data import get_loaders
from tqdm import tqdm

# Minimal BiasGPT wrapper replicating required statistics interface
class BiasGPT:
    def __init__(self, layer, metric_name):
        self.layer = layer
        self.metric_name = metric_name
        self.inp_list = []
        self.out_list = []
    def add_batch(self, inp, out):
        # inp: (B, hidden) or (B, hidden_size)
        self.inp_list.append(inp.detach())
        self.out_list.append(out.detach())
    @property
    def baseline_inp(self):
        if len(self.inp_list) == 0:
            return torch.zeros(1)
        return torch.mean(torch.cat(self.inp_list, dim=0), dim=0)
    @property
    def fluc_inp(self):
        if len(self.inp_list) == 0:
            return torch.zeros(1)
        x = torch.cat(self.inp_list, dim=0)
        return torch.var(x, dim=0)
    @property
    def scaler_inp(self):
        if len(self.inp_list) == 0:
            return torch.ones(1)
        x = torch.cat(self.inp_list, dim=0)
        return torch.var(x, dim=0) + 1e-8

metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_inp.reshape((1,-1)))).mean(axis=0),
}

def find_linear_layers(module, layers=[nn.Linear], name=''):
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_linear_layers(child, layers=layers, name=name + '.' + name1 if name != '' else name1))
    return res

def prepare_calibration_input(model, dataloader, device, nsamples):
    """Capture the inputs to the first transformer block for nsamples batches.
    Memory-reduced version: allocate only nsamples * seq_len * hidden instead of a large constant."""
    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers
    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]
    dtype = next(iter(model.parameters())).dtype
    seq_len = dataloader[0][0].shape[1]
    inps = torch.zeros((nsamples, seq_len, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, 'position_ids': None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            if cache['i'] < nsamples:
                inps[cache['i']] = inp[:, :seq_len]
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            cache['position_ids'] = kwargs.get('position_ids', None)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader[:nsamples]:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module
    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    model.config.use_cache = use_cache
    return inps, outs, attention_mask, position_ids

def compress_layer(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False):
    # Simplified: only apply masking (no structural shrink) for safety
    if attn_mask is not None:
        expanded = attn_mask.repeat_interleave(128).to(device)
        layer.self_attn.q_proj.weight.data *= expanded.unsqueeze(-1)
        layer.self_attn.k_proj.weight.data *= expanded.unsqueeze(-1)
        layer.self_attn.v_proj.weight.data *= expanded.unsqueeze(-1)
    if mlp_mask is not None:
        layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)
        layer.mlp.gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))

def prune_flap(args, model, tokenizer, device=torch.device("cuda:0")):
    # Provide default attributes if missing (minimal change integration)
    if not hasattr(args, 'pruning_ratio'):
        args.pruning_ratio = getattr(args, 'sparsity_ratio', 0.0)
    if not hasattr(args, 'metrics'):
        args.metrics = 'IFV'
    if not hasattr(args, 'structure'):
        args.structure = 'AL-AM'
    if not hasattr(args, 'remove_heads'):
        args.remove_heads = 0
    if not hasattr(args, 'unstr'):
        args.unstr = False
    use_cache = model.config.use_cache
    model.config.use_cache = False
    print("loading calibration data (FLAP)")
    dataset_name = getattr(args, 'prune_data', None)
    synthetic_only = False
    if not dataset_name or dataset_name == 'none':
        # Use synthetic calibration directly; skip expensive dataset loading
        synthetic_only = True
        dataset_name = 'synthetic'
    if synthetic_only:
        loaders = None
    else:
        try:
            loaders = get_loaders(dataset_name, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen, disentangle=True, tokenizer=tokenizer)
        except Exception as e:
            print(f"FLAP get_loaders failed for {dataset_name}: {e}. Using synthetic random data fallback.")
            loaders = None
    if loaders is None:
        # synthetic fallback
        print("Building synthetic calibration set")
        trainloader = []
        vocab_size = getattr(tokenizer, 'vocab_size', 32000)
        max_len = min(512, getattr(model, 'seqlen', 512))
        for _ in range(args.nsamples):
            inp = torch.randint(0, vocab_size, (1, max_len))
            tar = inp.clone(); tar[:, :-1] = -100
            trainloader.append((inp, tar))
        dataloader = trainloader
    else:
        dataloader, _ = loaders
    print(f"dataset loading complete (FLAP) using {dataset_name}; nsamples={len(dataloader)}")
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args.nsamples)
    layers = model.model.layers
    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []
    for i in tqdm(range(len(layers)), desc="FLAP Processing layers"):
        layer = layers[i]
        subset = {}
        flayers = find_linear_layers(layer)
        if 'self_attn.o_proj' in flayers and 'mlp.down_proj' in flayers:
            subset['self_attn.o_proj'] = flayers['self_attn.o_proj']
            subset['mlp.down_proj'] = flayers['mlp.down_proj']
        else:
            continue
        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics)
        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)
            return tmp
        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()
        for name in subset:
            if name == 'self_attn.o_proj':
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.half))
    if len(attn_metric_list) == 0:
        print("FLAP: no layers processed")
        return
    attn_metric_list = torch.cat(attn_metric_list)
    mlp_metric_list = torch.cat(mlp_metric_list)
    if args.structure in ["AL-AM", "AL-MM"]:
        attn_metric_list = attn_metric_list.reshape(len(layers), -1, 128).sum(dim=-1)
        mlp_metric_list = mlp_metric_list.reshape(len(layers), -1)
        attn_metric_list /= attn_metric_list.sum(dim=-1, keepdim=True)
        mlp_metric_list /= mlp_metric_list.sum(dim=-1, keepdim=True)
        layer_select = torch.arange(len(layers))
    else:  # UL-* structures
        attn_metric_list /= attn_metric_list.sum()
        mlp_metric_list /= mlp_metric_list.sum()
        layer_select = torch.tensor([0])
    attn_baseline_inp = torch.stack(attn_baseline_inp_list)
    mlp_baseline_inp = torch.stack(mlp_baseline_inp_list)
    # simple proportional pruning threshold
    attn_keep_ratio = 1 - args.pruning_ratio
    mlp_keep_ratio = 1 - args.pruning_ratio
    for idx in layer_select:
        heads = attn_metric_list[idx]
        neurons = mlp_metric_list[idx]
        k_heads = max(1, int(len(heads) * attn_keep_ratio))
        k_neurons = max(1, int(len(neurons) * mlp_keep_ratio))
        top_heads = torch.topk(heads, k_heads).indices
        top_neurons = torch.topk(neurons, k_neurons).indices
        head_mask = torch.zeros_like(heads, dtype=torch.bool)
        neuron_mask = torch.zeros_like(neurons, dtype=torch.bool)
        head_mask[top_heads] = True
        neuron_mask[top_neurons] = True
        compress_layer(layers[idx], head_mask, neuron_mask, attn_baseline_inp[idx], mlp_baseline_inp[idx], device, bias=True, unstr=args.unstr)
    model.config.use_cache = use_cache
    torch.cuda.empty_cache()
