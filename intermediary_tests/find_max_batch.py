"""
Binary-search the largest batch size that doesn't OOM, per T, per config
(linear+eager vs conv+compiled -- the two endpoint winners found so far).

Logs peak memory at each successful size too, so we can see whether memory
scales linearly with batch (expected) or something is blowing up unexpectedly
(e.g. compile/cudagraph buffer bloat, or leftover fragmentation).

Run on its own GPU, e.g.: CUDA_VISIBLE_DEVICES=2 python find_max_batch.py
"""
import gc
import json

import torch
import torch._dynamo
import yaml

import models
import datasets
import utils
from models.convs import set_pointwise_kind

CONFIG_PATH = "configs/test_one.yaml"
CHECKPOINT = "./save/recurrent_lte_paper_repro/epoch-best.pth"
BUDGETS = (16, 32, 64)
CONFIGS = [
    {"label": "linear+eager", "kind": "linear", "compiled": False},
    {"label": "conv+compiled", "kind": "conv", "compiled": True},
]
START_BATCH = 10000   # known-good starting point to grow from
MAX_CAP = 2_000_000   # don't search past this even if it never OOMs
OUT_JSON = "max_batch_results.json"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load(kind, compiled):
    set_pointwise_kind(kind)  # must precede models.make()
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get("data_norm"):
        utils.set_normalizer(**config["data_norm"])
    dataset = datasets.make(config["test_dataset"]["dataset"])
    dataset = datasets.make(config["test_dataset"]["wrapper"], args={"dataset": dataset})
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, num_workers=2)
    model = models.make(torch.load(CHECKPOINT, weights_only=False)["model"], load_sd=True).cuda().eval()
    inputs, _ = next(iter(loader))
    model.gen_feat(inputs["inp"].cuda())
    if compiled:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        model.predictor = torch.compile(model.predictor, dynamic=True)
    return model, inputs


def try_batch(model, coord_full, cell_full, batch, T):
    """Returns (ok, peak_mem_gb) -- tiles the real image's points if batch exceeds
    how many are actually available, since we just need realistic-shaped data."""
    n = coord_full.shape[1]
    reps = (batch + n - 1) // n
    coord = coord_full.repeat(1, reps, 1)[:, :batch]
    cell = cell_full.repeat(1, reps, 1)[:, :batch]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with torch.no_grad():
            model.query_t(coord, cell, T)
        torch.cuda.synchronize()
        peak = torch.cuda.max_memory_allocated() / 1e9
        return True, peak
    except torch.cuda.OutOfMemoryError:
        return False, None
    finally:
        del coord, cell
        gc.collect()
        torch.cuda.empty_cache()


def binary_search_max_batch(model, coord_full, cell_full, T):
    # grow until it breaks, to establish an upper bound
    lo = START_BATCH
    ok, peak = try_batch(model, coord_full, cell_full, lo, T)
    if not ok:
        return None, None  # breaks even at the known-good starting point

    hi = lo
    last_good_peak = peak
    while hi < MAX_CAP:
        hi *= 2
        ok, peak = try_batch(model, coord_full, cell_full, hi, T)
        if not ok:
            break
        lo, last_good_peak = hi, peak
    else:
        return hi, last_good_peak  # never broke even at MAX_CAP

    # binary search between lo (good) and hi (bad)
    while hi - lo > max(500, lo // 100):
        mid = (lo + hi) // 2
        ok, peak = try_batch(model, coord_full, cell_full, mid, T)
        if ok:
            lo, last_good_peak = mid, peak
        else:
            hi = mid
    return lo, last_good_peak


def main():
    results = {}
    for cfg in CONFIGS:
        model, inputs = load(cfg["kind"], cfg["compiled"])
        coord = inputs["coord"].cuda()
        cell = inputs["cell"].cuda()

        print(f"\n=== {cfg['label']} ===")
        for T in BUDGETS:
            max_batch, peak_gb = binary_search_max_batch(model, coord, cell, T)
            results[f"{cfg['label']}_T{T}"] = {"max_batch": max_batch, "peak_gb": peak_gb}
            if max_batch is None:
                print(f"  T={T:>2d}  BREAKS even at batch={START_BATCH}")
            else:
                print(f"  T={T:>2d}  max_batch={max_batch:>8d}  peak_mem={peak_gb:.2f} GB")

        del model
        gc.collect()
        torch.cuda.empty_cache()

    with open(OUT_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {OUT_JSON}")


if __name__ == "__main__":
    main()