import torch, time
import torch.nn.functional as F
from difficulty import compute_tile_difficulty, difficulty_to_budget, budget_map_to_coord_order
import utils

def run_config(model, inp, coord, cell, h, w, budgets=None, tile_size=8, fixed_num_pred=None, warmup=2, iters=10):
    model.gen_feat(inp)  # not timed — same cost in every config

    if fixed_num_pred is not None:
        model.predictor.num_pred = fixed_num_pred
        call = lambda: model.query(coord, cell)
    else:
        diff = compute_tile_difficulty(inp, tile_size=tile_size)
        bmap = difficulty_to_budget(diff, budgets=budgets)
        budget = budget_map_to_coord_order(bmap, h, w).to(inp.device)
        call = lambda: model.adaptive_query(coord, cell, budget)

    for _ in range(warmup):
        with torch.no_grad(): call()
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        with torch.no_grad(): out = call()
    torch.cuda.synchronize()
    avg_time = (time.time() - t0) / iters

    return out, avg_time

def evaluate(out, gt):
    recon = utils.denormalize(out['recon']).clamp_(0, 1)
    gt = utils.denormalize(gt).clamp_(0, 1)
    return utils.calc_psnr(recon, gt).item()