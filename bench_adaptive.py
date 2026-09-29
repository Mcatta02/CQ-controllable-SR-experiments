import os
import yaml
import torch
from types import SimpleNamespace
from torch.profiler import profile, ProfilerActivity

import models
import datasets
import utils
import test as test_mod


CONFIG_PATH = "configs/test_one.yaml"
CHECKPOINT = "./save/recurrent_lte_paper_repro/epoch-best.pth"
GPU = "3"

BUDGETS = [16, 32, 64]
METRIC = "sobel"
TILE_SIZE = 64
BATCH_SIZE = 100000


def load_model_and_data():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    if config.get("data_norm"):
        utils.set_normalizer(**config["data_norm"])

    dataset = datasets.make(config["test_dataset"]["dataset"])
    dataset = datasets.make(
        config["test_dataset"]["wrapper"],
        args={"dataset": dataset},
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=config["test_dataset"]["batch_size"],
        num_workers=16,
        pin_memory=True,
    )

    model_spec = torch.load(CHECKPOINT)["model"]
    model = models.make(model_spec, load_sd=True).cuda()
    model.eval()

    return model, loader, config


@torch.no_grad()
def profile_one_image(model, inputs, budgets):
    inp = inputs["inp"].cuda()
    coord = inputs["coord"].cuda()
    cell = inputs["cell"].cuda()

    # Same as batched_predict()
    model.gen_feat(inp)

    n = coord.shape[1]
    ih, iw = inp.shape[-2:]
    s = (n / (ih * iw)) ** 0.5
    out_h = round(ih * s)
    out_w = round(iw * s)

    diff = test_mod.compute_tile_difficulty(
        inp,
        tile_size=TILE_SIZE,
        metric=METRIC,
    )

    bmap = test_mod.difficulty_to_budget(
        diff,
        budgets=budgets,
    )

    budget = test_mod.budget_map_to_coord_order(
        bmap,
        out_h,
        out_w,
    ).to(inp.device)

    budget_flat = budget[0]

    print(f"Input:  {ih} x {iw}")
    print(f"Output: {out_h} x {out_w}")
    print(f"Points: {n:,}")

    print("\nBudget distribution:")
    for b in budget_flat.unique().tolist():
        count = (budget_flat == b).sum().item()
        print(f"  T={int(b):2d}: {count:,} points")

    # Warmup using EXACT same loop as batched_predict()
    for b in budget_flat.unique().tolist():
        idx = (budget_flat == b).nonzero(as_tuple=True)[0]

        ql = 0
        while ql < idx.shape[0]:
            qr = min(ql + BATCH_SIZE, idx.shape[0])
            sub_idx = idx[ql:qr]

            model.adaptive_query(
                coord[:, sub_idx],
                cell[:, sub_idx],
                budget[:, sub_idx],
            )

            ql = qr

    torch.cuda.synchronize()

    print("\nProfiling one full adaptive image...")

    with profile(
        activities=[
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:

        for b in budget_flat.unique().tolist():
            idx = (budget_flat == b).nonzero(as_tuple=True)[0]

            ql = 0
            while ql < idx.shape[0]:
                qr = min(ql + BATCH_SIZE, idx.shape[0])
                sub_idx = idx[ql:qr]

                model.adaptive_query(
                    coord[:, sub_idx],
                    cell[:, sub_idx],
                    budget[:, sub_idx],
                )

                ql = qr

        torch.cuda.synchronize()

    print("\n========== CUDA TIME ==========")
    print(
        prof.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=30,
        )
    )

    print("\n========== CPU TIME ==========")
    print(
        prof.key_averages().table(
            sort_by="cpu_time_total",
            row_limit=30,
        )
    )


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    model, loader, config = load_model_and_data()

    # First image only
    inputs, targets = next(iter(loader))

    profile_one_image(
        model,
        inputs,
        BUDGETS,
    )


if __name__ == "__main__":
    main()