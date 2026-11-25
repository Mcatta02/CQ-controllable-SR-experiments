"""
Quick timing probe used by compare_old_vs_new.sh. Works against EITHER the
pre-optimization baseline or the current optimized code -- it detects which
one is present (the baseline has no set_pointwise_kind/compile hook) and
behaves accordingly, so the same script measures both sides fairly.

Usage: python compare_timing.py --label OLD   (or NEW)
Writes <label.lower()>_results.json, and if both old_results.json and
new_results.json exist afterward, prints a side-by-side table.
"""
import argparse
import json
import os
import time

import torch
import yaml

import models
import datasets
import utils
import torch._dynamo

CONFIG_PATH = "configs/test_one.yaml"
CHECKPOINT = "./save/recurrent_lte_paper_repro/epoch-best.pth"
BATCH_SIZES = (10000, 100000)
BUDGETS = (4, 8, 16, 32, 64)
WARMUP = 5
ITERS = 10
CROSSOVER_BATCH = 25000  # only relevant for the optimized code

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def has_optimizations():
    try:
        from models.convs import set_pointwise_kind  # noqa: F401
        return True
    except ImportError:
        return False


def load(batch_size, optimized):
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get("data_norm"):
        utils.set_normalizer(**config["data_norm"])

    if optimized:
        from models.convs import set_pointwise_kind
        set_pointwise_kind("conv" if batch_size >= CROSSOVER_BATCH else "linear")

    dataset = datasets.make(config["test_dataset"]["dataset"])
    dataset = datasets.make(config["test_dataset"]["wrapper"], args={"dataset": dataset})
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    model = models.make(torch.load(CHECKPOINT, weights_only=False)["model"], load_sd=True).cuda().eval()
    inputs, _ = next(iter(loader))
    model.gen_feat(inputs["inp"].cuda())

    if optimized and batch_size >= CROSSOVER_BATCH:

        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        model.predictor = torch.compile(model.predictor, dynamic=True)

    return model, inputs


@torch.no_grad()
def time_query_t(model, coord, cell, T):
    for _ in range(WARMUP):
        model.query_t(coord, cell, T)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(ITERS):
        model.query_t(coord, cell, T)
    torch.cuda.synchronize()
    return (time.time() - t0) / ITERS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True, choices=["OLD", "NEW"])
    args = parser.parse_args()

    optimized = has_optimizations()
    print(f"[{args.label}] optimizations detected in models/convs.py: {optimized}")

    results = {}
    for batch in BATCH_SIZES:
        model, inputs = load(batch, optimized)
        coord = inputs["coord"].cuda()[:, :batch]
        cell = inputs["cell"].cuda()[:, :batch]
        for T in BUDGETS:
            t = time_query_t(model, coord, cell, T)
            print(f"  batch={batch:>7d}  T={T:>2d}  {t * 1e3:8.2f} ms/call")
            results[f"batch{batch}_T{T}"] = t * 1e3
        del model
        torch.cuda.empty_cache()

    out_path = f"{args.label.lower()}_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {out_path}")

    old_path, new_path = "old_results.json", "new_results.json"
    if os.path.exists(old_path) and os.path.exists(new_path):
        with open(old_path) as f:
            old = json.load(f)
        with open(new_path) as f:
            new = json.load(f)
        print("\n=== OLD vs NEW (ms/call) ===")
        print(f"{'config':20s} {'old':>12s} {'new':>12s} {'speedup':>10s}")
        for key in old:
            if key in new:
                speedup = old[key] / new[key]
                print(f"{key:20s} {old[key]:12.2f} {new[key]:12.2f} {speedup:9.2f}x")


if __name__ == "__main__":
    main()
