"""
Correctness check: compiled predictor vs eager predictor (NOT old-vs-new code --
that's parity_check.py). Run standalone, no "save" step needed since eager is
computed fresh in the same run as the ground truth.

"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import yaml
import torch
import torch._dynamo

import models
import datasets
import utils
from difficulty import compute_tile_difficulty, difficulty_to_budget, budget_map_to_coord_order

CONFIG_PATH = "configs/test_one.yaml"
CHECKPOINT = "./save/recurrent_lte_paper_repro/epoch-best.pth"
N_POINTS = 4000
BUDGETS = [16, 32, 64]
METRIC, TILE_SIZE = "sobel", 64
TOL = 1e-3

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
    model = models.make(torch.load(CHECKPOINT, weights_only=False)["model"], load_sd=True).cuda().eval()
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
    for T in (16, 32, 64):
        torch.manual_seed(0)
        outs[f"query_t_{T}"] = model.query_t(c, ce, T).cpu()
    torch.manual_seed(0)
    outs["adaptive"] = model.adaptive_query(c, ce, b)["recon"].cpu()
    return outs


def main():
    model, inputs = load()

    torch.manual_seed(0)
    eager_out_seed0 = run(model, inputs)
    torch.manual_seed(1)
    eager_out_seed1 = run(model, inputs)
    print("--- sanity check: eager vs eager, different seeds ---")
    print("(if these diffs are already ~1e-2, the model's positional encoding is")
    print(" inherently stochastic per-call, and the compiled-vs-eager FAIL below")
    print(" is expected RNG divergence, not a correctness bug)")
    for k, v in eager_out_seed0.items():
        d = (v - eager_out_seed1[k]).abs().max().item()
        print(f"     {k:12s} max abs diff (seed 0 vs seed 1) = {d:.3e}")

    eager_out = eager_out_seed0

    # NOTE: must reuse the SAME model object -- gen_feat's cached self.feat must
    # match, and query_t/adaptive_query set self.predictor.num_pred as a side
    # effect that torch.compile guards on, so compiling in place (not a fresh
    # reload) is exactly what production code will actually do.
    torch._dynamo.config.cache_size_limit = 64
    model.predictor = torch.compile(model.predictor, backend="aot_eager", dynamic=False)    
    torch.manual_seed(0)
    compiled_out = run(model, inputs)

    print("\n--- eager (seed 0) vs compiled (seed 0) ---")
    ok = True
    for k, v in eager_out.items():
        d = (v - compiled_out[k]).abs().max().item()
        good = d < TOL
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {k:12s} max abs diff (eager vs compiled) = {d:.3e}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()