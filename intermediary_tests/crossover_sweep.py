"""
Sweeps a range of batch sizes to find where linear+eager stops winning and
conv+compiled starts winning (or vice versa), per T.

Run on its own GPU, e.g.: CUDA_VISIBLE_DEVICES=3 python crossover_sweep.py

Edit BATCH_SIZES below once find_max_batch.py tells you the real ceiling per T --
no point testing batch sizes that would OOM anyway.
"""
import csv
import time

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
BATCH_SIZES = (5000, 10000, 20000, 30000, 50000, 75000, 100000)
CONFIGS = [
    {"label": "linear+eager", "kind": "linear", "compiled": False},
    {"label": "conv+compiled", "kind": "conv", "compiled": True},
]
WARMUP = 5
ITERS = 10
OUT_CSV = "crossover_results.csv"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load(kind, compiled):
    set_pointwise_kind(kind)
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


def tile_to(coord_full, cell_full, batch):
    n = coord_full.shape[1]
    reps = (batch + n - 1) // n
    return coord_full.repeat(1, reps, 1)[:, :batch], cell_full.repeat(1, reps, 1)[:, :batch]


def main():
    rows = []
    for cfg in CONFIGS:
        model, inputs = load(cfg["kind"], cfg["compiled"])
        coord_full = inputs["coord"].cuda()
        cell_full = inputs["cell"].cuda()

        print(f"\n=== {cfg['label']} ===")
        for batch in BATCH_SIZES:
            coord, cell = tile_to(coord_full, cell_full, batch)
            for T in BUDGETS:
                try:
                    t = time_query_t(model, coord, cell, T)
                    print(f"  batch={batch:>7d}  T={T:>2d}  {t * 1e3:8.2f} ms/call")
                    rows.append({"config": cfg["label"], "batch": batch, "T": T, "ms_per_call": t * 1e3})
                except torch.cuda.OutOfMemoryError:
                    print(f"  batch={batch:>7d}  T={T:>2d}  OOM")
                    rows.append({"config": cfg["label"], "batch": batch, "T": T, "ms_per_call": None})
                    torch.cuda.empty_cache()

        del model, coord_full, cell_full
        torch.cuda.empty_cache()

    with open(OUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "batch", "T", "ms_per_call"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()