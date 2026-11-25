#!/usr/bin/env python3
"""Extra CQ-SR experiments: output agreement and adaptive component allocation.

Run from the accelerated repository root in the recurrent_lte environment.
The released checkpoint is loaded independently for original and accelerated paths.
"""
import os as _os, sys as _sys  # run from repo root: python tests/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import argparse
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_model(root, ckpt, mode, bsize, no_compile=False):
    import sys
    sys.path.insert(0, str(Path(root).expanduser().resolve()))
    os.chdir(Path(root).expanduser())
    os.environ.setdefault('POINTWISE_KIND', 'linear')
    if no_compile:
        def _identity_compile(fn=None, **kwargs):
            return (lambda f: f) if fn is None else fn
        torch.compile = _identity_compile
    import models
    ck = torch.load(str(Path(ckpt).expanduser()), map_location='cpu')
    model = models.make(ck['model'], load_sd=True).cuda().eval()
    if mode == 'fast':
        from models.fast_recurrent import install
        from models.fast_post import install_fast_post
        install(model, batch_size=bsize)
        try:
            install_fast_post(model, compile_post=not no_compile)
        except TypeError:
            install_fast_post(model)
    return model


def make_dataset(data_root, n_images, scale):
    import datasets, utils
    utils.set_normalizer([0.5] * 3, [0.5] * 3)
    ds = datasets.make({'name': 'image-folder', 'args': {
        'root_path': str(Path(data_root).expanduser()), 'first_k': n_images}})
    return datasets.make({'name': 'sr-implicit-downsampled', 'args': {
        'scale_min': float(scale), 'scale_max': float(scale)}}, args={'dataset': ds})


def get_image(ds, i):
    inputs, targets = ds[i]
    return ({k: v.unsqueeze(0).cuda() for k, v in inputs.items()},
            {k: v.unsqueeze(0).cuda() for k, v in targets.items()})


def set_T(model, T):
    model.predictor.num_pred = int(T)
    model.num_pred = int(T)
    model.num_preds = int(T)


def predict_fixed(model, inputs, batch_size):
    coord, cell = inputs['coord'], inputs['cell']
    model.gen_feat(inputs['inp'])
    pieces = []
    with torch.no_grad():
        for start in range(0, coord.shape[1], batch_size):
            end = min(start + batch_size, coord.shape[1])
            pieces.append(model.query(coord[:, start:end], cell[:, start:end])['recon'].cpu())
    return torch.cat(pieces, dim=1)


def image_metrics(pred_flat, inputs, targets, lpips_net):
    import utils
    ih, iw = inputs['inp'].shape[-2:]
    n = inputs['coord'].shape[1]
    s = math.sqrt(n / (ih * iw))
    pred = pred_flat.cuda().view(1, round(ih * s), round(iw * s), -1)
    pred = pred.permute(0, 3, 1, 2).contiguous()
    pred = pred[..., :targets['gt_img'].shape[-2], :targets['gt_img'].shape[-1]]
    pred_raw = utils.denormalize(pred)
    pred01 = pred_raw.clamp(0, 1)
    gt01 = utils.denormalize(targets['gt_img']).clamp(0, 1)
    # Keep the same 8-bit discretisation convention as the existing evaluation.
    pred_disc = pred01.clone()
    for b in range(pred_disc.shape[0]):
        pred_disc[b] = utils.discretize(pred_disc[b])
    pred_norm = utils.normalize(pred_disc)
    lp = float(lpips_net(pred_norm, targets['gt_img']).mean().item())
    ps = float(utils.calc_psnr(utils.denormalize(pred_norm).clamp(0, 1), gt01, dataset='div2k', scale=4))
    return pred_raw, pred01, ps, lp


def cmd_equiv(args):
    from lpips import LPIPS
    ds = make_dataset(args.data_root, args.images, args.scale)
    lpips_net = LPIPS(net='alex').cuda().eval()
    outdir = Path(args.out).expanduser(); outdir.mkdir(parents=True, exist_ok=True)
    models = {
        'original': load_model(args.root, args.ckpt, 'original', args.batch_size),
        'fast': load_model(args.root, args.ckpt, 'fast', args.batch_size, args.no_compile),
    }
    # Warm up each implementation/budget before collecting timings.
    warm_inputs, _ = get_image(ds, 0)
    for T in args.Ts:
        for impl, model in models.items():
            set_T(model, T)
            seed_all(12345)
            _ = predict_fixed(model, warm_inputs, args.batch_size)
            torch.cuda.synchronize()
    rows = []
    for T in args.Ts:
        for impl, model in models.items():
            set_T(model, T)
        for i in range(min(args.images, len(ds))):
            inputs, targets = get_image(ds, i)
            for seed in args.seeds:
                outputs = {}
                qualities = {}
                for impl, model in models.items():
                    seed_all(seed)
                    torch.cuda.synchronize(); t0 = time.perf_counter()
                    flat = predict_fixed(model, inputs, args.batch_size)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
                    pred_raw, pred01, psnr, lpips = image_metrics(flat, inputs, targets, lpips_net)
                    outputs[impl] = pred_raw.cpu()
                    qualities[impl] = (psnr, lpips, elapsed)
                a, b = outputs['original'], outputs['fast']
                d = (a - b).abs()
                row = {
                    'T': T, 'image': i, 'seed': seed,
                    'max_abs': float(d.max()), 'mean_abs': float(d.mean()),
                    'rmse': float(torch.mean((a-b)**2).sqrt()),
                    'p99_abs': float(torch.quantile(d.flatten(), 0.99)),
                    'frac_gt_1e-6': float((d > 1e-6).float().mean()),
                    'frac_gt_1e-5': float((d > 1e-5).float().mean()),
                    'frac_gt_1e-4': float((d > 1e-4).float().mean()),
                    'psnr_orig': qualities['original'][0], 'psnr_fast': qualities['fast'][0],
                    'lpips_orig': qualities['original'][1], 'lpips_fast': qualities['fast'][1],
                    'time_orig_s': qualities['original'][2], 'time_fast_s': qualities['fast'][2],
                }
                rows.append(row)
                np.savez_compressed(outdir / f'equiv_T{T}_img{i:03d}_seed{seed}.npz',
                                    original=a.numpy(), fast=b.numpy(), diff=d.numpy())
                print(json.dumps(row), flush=True)
    with open(outdir / 'equivalence.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    summary = {}
    for T in args.Ts:
        rr = [r for r in rows if r['T'] == T]
        summary[str(T)] = {k: {'mean': float(np.mean([r[k] for r in rr])),
                               'max': float(np.max([r[k] for r in rr]))}
                          for k in ('max_abs', 'mean_abs', 'rmse', 'p99_abs', 'psnr_orig', 'psnr_fast')}
    (outdir / 'equivalence_summary.json').write_text(json.dumps(summary, indent=2))
    print('Saved:', outdir)


def tile_scores(inp, tile_size, metric):
    # inp is normalized RGB [1,3,H,W]. The scale factor does not affect ranking.
    x = inp.mean(dim=1, keepdim=True)
    if metric == 'sobel':
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]], device=x.device).view(1,1,3,3)
        ky = kx.transpose(-1, -2)
        gx = F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'), kx)
        gy = F.conv2d(F.pad(x, (1,1,1,1), mode='replicate'), ky)
        score = torch.sqrt(gx.square() + gy.square() + 1e-12)
    elif metric == 'variance':
        score = x
    else:
        raise ValueError(metric)
    _, _, H, W = score.shape
    rows = math.ceil(H / tile_size); cols = math.ceil(W / tile_size)
    vals, weights = [], []
    for r in range(rows):
        for c in range(cols):
            patch = score[:, :, r*tile_size:min((r+1)*tile_size,H), c*tile_size:min((c+1)*tile_size,W)]
            if metric == 'variance':
                val = patch.var(unbiased=False)
            else:
                val = patch.mean()
            vals.append(val.reshape(()))
            weights.append(patch.shape[-2] * patch.shape[-1])
    return torch.stack(vals), torch.tensor(weights, device=inp.device), rows, cols


def make_budget_map(inp, coord, tile_size, budgets, strategy, seed):
    scores, _, rows, cols = tile_scores(inp, tile_size, 'sobel' if strategy == 'sobel' else 'variance')
    H, W = inp.shape[-2:]
    # Query-to-nearest-LR-pixel mapping matching grid_sample(..., mode='nearest', align_corners=False).
    xy = coord[0]
    ix = torch.floor((xy[:, 0] + 1.0) * W / 2.0).long().clamp(0, W-1)
    iy = torch.floor((xy[:, 1] + 1.0) * H / 2.0).long().clamp(0, H-1)
    tile_id = (iy // tile_size).clamp(max=rows-1) * cols + (ix // tile_size).clamp(max=cols-1)
    tile_query_counts = torch.bincount(tile_id, minlength=rows*cols).float()
    if strategy == 'random':
        g = torch.Generator(device='cpu'); g.manual_seed(int(seed))
        scores = torch.rand(scores.shape, generator=g).to(scores.device)
    # Sort tiles by score; assign four budgets by cumulative QUERY count, not tile count.
    order = torch.argsort(scores)
    counts_sorted = tile_query_counts[order]
    cumulative = counts_sorted.cumsum(0)
    total = max(float(cumulative[-1]), 1.0)
    thresholds = torch.tensor([0.25, 0.50, 0.75], device=inp.device) * total
    rank = torch.searchsorted(thresholds, cumulative, right=False).clamp(max=3)
    budget_by_tile = torch.empty(rows*cols, device=inp.device, dtype=torch.long)
    budget_values = torch.tensor(budgets, device=inp.device, dtype=torch.long)
    budget_by_tile[order] = budget_values[rank]
    # Map each output query to the budget of its nearest LR feature pixel.
    return budget_by_tile[tile_id].view(1, -1)


def predict_adaptive(model, inputs, batch_size, tile_size, budgets, strategy, seed):
    coord, cell, inp = inputs['coord'], inputs['cell'], inputs['inp']
    model.gen_feat(inp)
    budget = make_budget_map(inp, coord, tile_size, budgets, strategy, seed)
    flat_budget = budget[0]
    recon = torch.empty((1, coord.shape[1], model.decoder.out_dim), device=inp.device, dtype=inp.dtype)
    with torch.no_grad():
        for b in torch.unique(flat_budget).tolist():
            idx = torch.nonzero(flat_budget == b, as_tuple=True)[0]
            for start in range(0, idx.numel(), batch_size):
                sub = idx[start:start+batch_size]
                result = model.adaptive_query(coord[:, sub], cell[:, sub], budget[:, sub])
                recon[:, sub] = result['recon'].detach()
    return recon.cpu(), budget.cpu()


def cmd_adaptive(args):
    from lpips import LPIPS
    import utils
    ds = make_dataset(args.data_root, args.images, args.scale)
    lpips_net = LPIPS(net='alex').cuda().eval()
    model = load_model(args.root, args.ckpt, 'fast', args.batch_size, args.no_compile)
    outdir = Path(args.out).expanduser(); outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    strategies = ['fixed', 'random', 'sobel'] + (['variance'] if args.include_variance else [])
    # Warm up fixed and adaptive paths, including each recurrent budget bucket.
    warm_inputs, _ = get_image(ds, 0)
    for strategy in strategies:
        seed_all(12345)
        if strategy == 'fixed':
            set_T(model, args.fixed_T)
            _ = predict_fixed(model, warm_inputs, args.batch_size)
        else:
            _ = predict_adaptive(model, warm_inputs, args.batch_size, args.tile_size,
                                 args.budgets, strategy, 12345)
        torch.cuda.synchronize()
    for i in range(min(args.images, len(ds))):
        inputs, targets = get_image(ds, i)
        for seed in args.seeds:
            for strategy in strategies:
                if strategy == 'fixed':
                    set_T(model, args.fixed_T)
                    seed_all(seed); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
                    flat = predict_fixed(model, inputs, args.batch_size)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
                    budget = torch.full((1, inputs['coord'].shape[1]), args.fixed_T, dtype=torch.long)
                else:
                    seed_all(seed); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize(); t0 = time.perf_counter()
                    flat, budget = predict_adaptive(model, inputs, args.batch_size, args.tile_size,
                                                    args.budgets, strategy, seed)
                    torch.cuda.synchronize(); elapsed = time.perf_counter() - t0
                pred_raw, pred01, psnr, lpips = image_metrics(flat, inputs, targets, lpips_net)
                vals, counts = torch.unique(budget, return_counts=True)
                dist = {str(int(v)): int(c) for v, c in zip(vals, counts)}
                row = {'image': i, 'seed': seed, 'strategy': strategy,
                       'mean_T_query_weighted': float(budget.float().mean()),
                       'runtime_s_including_allocation': elapsed,
                       'peak_gpu_mb': torch.cuda.max_memory_allocated() / (1024**2),
                       'psnr': psnr, 'lpips': lpips,
                       'budget_counts': json.dumps(dist, sort_keys=True)}
                rows.append(row); print(json.dumps(row), flush=True)
    with open(outdir / 'adaptive.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    # Aggregate means and standard deviations across image/seed observations.
    summary = {}
    for strategy in strategies:
        rr = [r for r in rows if r['strategy'] == strategy]
        summary[strategy] = {k: {'mean': float(np.mean([r[k] for r in rr])),
                                 'std': float(np.std([r[k] for r in rr], ddof=1)) if len(rr)>1 else 0.0}
                             for k in ('mean_T_query_weighted', 'runtime_s_including_allocation', 'peak_gpu_mb', 'psnr', 'lpips')}
    (outdir / 'adaptive_summary.json').write_text(json.dumps(summary, indent=2))
    print('Saved:', outdir)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument('--root', default='~/CQ-controllable-SR-experiments')
    common.add_argument('--ckpt', default='~/CQ-orig/base.pth')
    common.add_argument('--data-root', default='~/CQ-controllable-SR-experiments/load/div2k/DIV2K_valid_HR')
    common.add_argument('--scale', type=float, default=4)
    common.add_argument('--images', type=int, default=10)
    common.add_argument('--batch-size', type=int, default=150000)
    common.add_argument('--no-compile', action='store_true')
    common.add_argument('--seeds', type=int, nargs='+', default=[0,1,2])
    common.add_argument('--out', default='./extra_test_results')
    e = sub.add_parser('equiv', parents=[common])
    e.add_argument('--Ts', type=int, nargs='+', default=[4,16,64])
    e.set_defaults(func=cmd_equiv)
    a = sub.add_parser('adaptive', parents=[common])
    a.add_argument('--tile-size', type=int, default=16)
    a.add_argument('--budgets', type=int, nargs='+', default=[8,16,32,64])
    a.add_argument('--fixed-T', type=int, default=30)
    a.add_argument('--include-variance', action='store_true')
    a.set_defaults(images=20, func=cmd_adaptive)
    args = p.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
