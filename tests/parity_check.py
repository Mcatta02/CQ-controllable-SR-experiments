"""
Parity check for the step-1 refactor (None-init state, baddbmm, Linear layers, no per-step cat).

  1) BEFORE copying the patched files in:   python parity_check.py save
  2) AFTER copying them in:                 python parity_check.py compare

Runs the real checkpoint on a strided slice of one image (hits every budget) through
query_t (T = 4, 16, 64) and adaptive_query, with the RNG re-seeded before each call so the
stochastic position codes match, and TF32 off so differences are only fp32 rounding.
"""
import os
import sys

GPU = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = GPU

import yaml
import torch

import models
import datasets
import utils
from difficulty import compute_tile_difficulty, difficulty_to_budget, budget_map_to_coord_order

CONFIG_PATH = "configs/test_one.yaml"
CHECKPOINT = "./save/recurrent_lte_paper_repro/epoch-best.pth"
REF_PATH = "parity_ref.pt"
N_POINTS = 4000
BUDGETS = [16, 32, 64]
METRIC, TILE_SIZE = "sobel", 64
TOL = 1e-3  # max abs difference allowed on the (normalized) reconstruction

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False


def load():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get("data_norm"):
        utils.set_normalizer(**config["data_norm"])
    dataset = datasets.make(config["test_dataset"]["dataset"])
    dataset = datasets.make(config["test_dataset"]["wrapper"], args={"dataset": dataset})
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    model = models.make(torch.load(CHECKPOINT)["model"], load_sd=True).cuda().eval()
    inputs, _ = next(iter(loader))
    return model, inputs


@torch.no_grad()
def run(model, inputs):
    inp = inputs["inp"].cuda()
    coord = inputs["coord"].cuda()
    cell = inputs["cell"].cuda()
    model.gen_feat(inp)

    n = coord.shape[1]
    ih, iw = inp.shape[-2:]
    s = (n / (ih * iw)) ** 0.5
    out_h, out_w = round(ih * s), round(iw * s)
    bmap = difficulty_to_budget(
        compute_tile_difficulty(inp, tile_size=TILE_SIZE, metric=METRIC), budgets=BUDGETS)
    budget = budget_map_to_coord_order(bmap, out_h, out_w).to(inp.device)

    idx = torch.linspace(0, n - 1, N_POINTS, device=inp.device).long()
    c, ce, b = coord[:, idx], cell[:, idx], budget[:, idx]

    outs = {}
    for T in (4, 16, 64):
        torch.manual_seed(0)
        outs[f"query_t_{T}"] = model.query_t(c, ce, T).cpu()
    torch.manual_seed(0)
    outs["adaptive"] = model.adaptive_query(c, ce, b)["recon"].cpu()
    return outs


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    model, inputs = load()
    outs = run(model, inputs)
    if mode == "save":
        torch.save(outs, REF_PATH)
        print(f"saved reference outputs to {REF_PATH}")
    elif mode == "compare":
        ref = torch.load(REF_PATH)
        ok = True
        for k, v in outs.items():
            d = (v - ref[k]).abs().max().item()
            good = d < TOL
            ok &= good
            print(f"{'OK  ' if good else 'FAIL'} {k:12s} max abs diff = {d:.3e}")
        sys.exit(0 if ok else 1)
    else:
        print(__doc__)


if __name__ == "__main__":
    main()