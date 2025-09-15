import os
import sys
import math
import shutil
import tempfile
import torch
import torch.nn as nn

# Ensure we can import lib.prune
CUR_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(CUR_DIR, '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from lib.prune import prune_wanda  # noqa: E402
import lib.prune as prune_mod  # noqa: E402


class DummyLayer(nn.Module):
    def __init__(self, in_dim=8, out_dim=8):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)

    def forward(self, x, attention_mask=None, position_ids=None):  # match expected signature
        return self.linear(x), None


class DummyModel(nn.Module):
    class Inner(nn.Module):
        def __init__(self, n_layers, in_dim, out_dim):
            super().__init__()
            self.layers = nn.ModuleList([DummyLayer(in_dim, out_dim) for _ in range(n_layers)])

    def __init__(self, n_layers=1, in_dim=8, out_dim=8, seqlen=4):
        super().__init__()
        self.config = type("cfg", (object,), {"use_cache": False})()
        self.model = DummyModel.Inner(n_layers, in_dim, out_dim)
        self.seqlen = seqlen
        self.hf_device_map = {}  # empty device map -> stays on CPU

    def forward(self, x):  # minimal forward used only during calibration capture
        attention_mask = torch.ones(1, self.seqlen, dtype=torch.long, device=x.device)
        position_ids = torch.arange(0, self.seqlen, dtype=torch.long, device=x.device).unsqueeze(0)
        out = x
        for layer in self.model.layers:
            out, _ = layer(out, attention_mask=attention_mask, position_ids=position_ids)
        return out


class Args:
    def __init__(self, **kwargs):
        # Provide defaults for all accessed attributes in prune_wanda
        self.use_diff = False
        self.recover_from_base = False
        self.neg_prune = False
        self.prune_part = False
        self.disentangle = False
        self.nsamples = 1
        self.seed = 42
        self.sparsity_ratio = 0.5  # not used when dump_wanda_score=True
        self.dump_wanda_score = True
        self.use_variant = False
        self.save = None
        # allow overrides
        for k, v in kwargs.items():
            setattr(self, k, v)


def make_args(save_dir, **overrides):
    return Args(save=save_dir, **overrides)


def fake_get_loaders_factory(tensor_store, hidden_size):
    def fake_get_loaders(prune_data, nsamples, seed, seqlen, tokenizer, disentangle):
        torch.manual_seed(seed)
        batches = []
        for i in range(nsamples):
            # (1, seqlen, hidden_size)
            inp = torch.randn(1, seqlen, hidden_size)
            tar = torch.zeros(1, seqlen, dtype=torch.long)  # keep all tokens (no -100)
            tensor_store.append(inp.clone())
            batches.append((inp, tar))
        return batches, None
    return fake_get_loaders


def compute_expected_w_metric(weight, inputs):
    # inputs: list of tensors shape (1, seqlen, hidden)
    # scaler_row after one sample = norm^2 over tokens for each hidden dim
    inp = inputs[0].reshape(-1, inputs[0].shape[-1])  # (tokens, hidden)
    scaler_row = torch.norm(inp, p=2, dim=0) ** 2  # (hidden,)
    act = torch.sqrt(scaler_row).reshape(1, -1)  # (1, hidden)
    magnitude = weight.abs()
    return magnitude * act  # broadcast on rows


def compute_expected_w_metric_diff(weight, weight_base, inputs):
    inp = inputs[0].reshape(-1, inputs[0].shape[-1])
    scaler_row = torch.norm(inp, p=2, dim=0) ** 2
    act = torch.sqrt(scaler_row).reshape(1, -1)
    magnitude = (weight - weight_base).abs()
    return magnitude * act


def run_single_case(prune_data, use_diff=False, disentangle=False):
    tmpdir = tempfile.mkdtemp(prefix="wanda_test_")
    try:
        # Build models
        model = DummyModel()
        model_base = None
        if use_diff:
            model_base = DummyModel()
            # Make base model weights slightly different so diff is non-zero
            with torch.no_grad():
                for (p, pb) in zip(model.parameters(), model_base.parameters()):
                    pb.copy_(p + 0.1)

        # Prepare args
        args = make_args(
            tmpdir,
            use_diff=use_diff,
            disentangle=disentangle,
            nsamples=1,
            dump_wanda_score=True,
        )

        # Monkeypatch get_loaders
        tensor_store = []
        fake_loader = fake_get_loaders_factory(tensor_store, hidden_size=8)
        original_get_loaders = prune_mod.get_loaders
        prune_mod.get_loaders = fake_loader

        try:
            prune_wanda(
                args=args,
                model=model,
                tokenizer=None,
                model_base=model_base,
                device=torch.device("cpu"),
                prune_data=prune_data,
            )
        finally:
            prune_mod.get_loaders = original_get_loaders

        # Determine expected folder and file names
        suffix = "weight_only"
        if use_diff:
            suffix = "weight_diff"
        if disentangle:
            suffix += "_disentangle"
        save_folder = os.path.join(tmpdir, f"wanda_score/{prune_data}_{suffix}")
        assert os.path.isdir(save_folder), f"Expected folder missing: {save_folder}"

        # There should be exactly one layer and one linear => one file pair (.pkl placeholder + _torch.pt)
        pt_files = [f for f in os.listdir(save_folder) if f.endswith("_torch.pt")]
        assert len(pt_files) == 1, f"Expected 1 wanda score pt file, found {pt_files}"
        pt_path = os.path.join(save_folder, pt_files[0])
        assert os.path.isfile(pt_path), "Wanda score tensor file missing"
        W_metric_saved = torch.load(pt_path)

        # Compute expected
        weight = next(model.model.layers[0].linear.parameters()).detach()
        if use_diff:
            weight_base = next(model_base.model.layers[0].linear.parameters()).detach()
            expected = compute_expected_w_metric_diff(weight, weight_base, tensor_store)
        else:
            expected = compute_expected_w_metric(weight, tensor_store)

        assert W_metric_saved.shape == expected.shape, "Shape mismatch for W_metric"
        max_abs_err = (W_metric_saved - expected).abs().max().item()
        assert max_abs_err < 1e-5, f"W_metric values differ (max abs err {max_abs_err})"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_weight_only():
    run_single_case(prune_data="testdata", use_diff=False, disentangle=False)


def test_weight_only_disentangle():
    run_single_case(prune_data="testdata2", use_diff=False, disentangle=True)


def test_weight_diff():
    run_single_case(prune_data="testdata3", use_diff=True, disentangle=False)


def test_weight_diff_disentangle():
    run_single_case(prune_data="testdata4", use_diff=True, disentangle=True)


if __name__ == "__main__":
    # Run tests manually without pytest
    test_weight_only()
    test_weight_only_disentangle()
    test_weight_diff()
    test_weight_diff_disentangle()
    print("All prune_wanda Wanda score tests passed.")
