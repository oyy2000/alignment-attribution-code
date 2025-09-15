import os
import tempfile
import shutil
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from lib.prune import prune_wanda


class Args:
    def __init__(self):
        self.use_diff = False
        self.recover_from_base = False
        self.neg_prune = False
        self.prune_part = False
        self.disentangle = False
        self.nsamples = 1
        self.seed = 42
        self.sparsity_ratio = 0.5
        self.dump_wanda_score = True  # only dump scores
        self.use_variant = False
        self.save = None


def test_prune_wanda_wikitext_real():
    tmpdir = tempfile.mkdtemp(prefix="wanda_real_")
    try:
        args = Args()
        args.save = tmpdir
        model_name = "sshleifer/tiny-gpt2"  # very small model for fast test
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Attach attributes expected by prune_wanda
        model.seqlen = 64
        model.hf_device_map = {}
        model.to(torch.device("cpu"))

        prune_wanda(
            args=args,
            model=model,
            tokenizer=tokenizer,
            model_base=None,
            device=torch.device("cpu"),
            prune_data="wikitext",
        )

        # Check that some wanda_score folder got created
        base_folder = os.path.join(tmpdir, "wanda_score")
        assert os.path.isdir(base_folder), "wanda_score folder not created"
        # Find subfolder for wikitext weight_only
        subfolders = [d for d in os.listdir(base_folder) if d.startswith("wikitext_weight_only")]
        assert subfolders, "Expected wikitext_weight_only* folder"
        target = os.path.join(base_folder, subfolders[0])
        pt_files = [f for f in os.listdir(target) if f.endswith("_torch.pt")]
        assert pt_files, "No wanda score tensor files saved"
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    test_prune_wanda_wikitext_real()
    print("Real-data prune_wanda integration test passed.")
