#!/usr/bin/env python
"""
Extra worker commands (one process = one implementation = one JSON result), same conventions as worker.py:

    python suite/worker_extra.py <scal|noise_equiv|arrange_exact|arrange_stat|adapt_vis> <spec.json>

  scal           time / peak memory / OOM against LR size, queries per call and T (synthetic LR sizes), + sharing stats
  noise_equiv    one call on a block of queries with a fixed seed; dumps generator outputs per iteration, phase head
                 outputs and RGB so that the original and the accelerated implementation can be compared (extra_report.py)
  arrange_exact  correctness of sharing under different query arrangements, with the SAME noise in every call
                 (the RNG is re-seeded before every model.query call, so any arrangement must give the same colours)
  arrange_stat   full images, scales and batch sizes: order (grid / shuffled / pixel-mates split), sharing ratio, time, PSNR
  adapt_vis      reconstructions, budget maps and fixed-T references for the local-quality analysis of adaptive inference
"""
import json
import math
import os
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import worker as W  # noqa: E402  (stdlib only at import time)


# ------------------------------------------------------------------ helpers
def pixel_ids(torch, lr_hw, coord):
    """Nearest LR pixel of every query (coord [Q,2] in (y,x), [-1,1]); same rule as worker.sharing_stats."""
    ih, iw = lr_hw
    iy = ((coord[:, 0] + 1) * ih / 2).floor().long().clamp_(0, ih - 1)
    ix = ((coord[:, 1] + 1) * iw / 2).floor().long().clamp_(0, iw - 1)
    return iy * iw + ix


def rollouts_for_order(ids_in_order, chunk, torch):
    """Number of rollouts the shared implementation runs when queries are processed in this order, chunk by chunk."""
    return sum(int(torch.unique(ids_in_order[l:l + chunk]).numel()) for l in range(0, ids_in_order.numel(), chunk))


def grid_hw(inputs):
    ih, iw = inputs["inp"].shape[-2:]
    s = math.sqrt(inputs["coord"].shape[1] / (ih * iw))
    return round(ih * s), round(iw * s)


def central_block(inputs, n_q):
    """Consecutive full HR rows around the image centre (so queries of one LR pixel are together)."""
    H, Wd = grid_hw(inputs)
    rows = max(1, n_q // Wd)
    r0 = (H - rows) // 2
    return slice(r0 * Wd, (r0 + rows) * Wd), rows, Wd


def flat_to_img(recon_flat, inputs, targets, utils):
    """[1,Q,3] normalised -> [3,h,w] in [0,1], cropped to the ground truth (as in the metrics code)."""
    H, Wd = grid_hw(inputs)
    r = recon_flat.cuda().view(1, H, Wd, -1).permute(0, 3, 1, 2).contiguous()
    r = r[..., :targets["gt_img"].shape[-2], :targets["gt_img"].shape[-1]]
    return utils.denormalize(r).clamp_(0, 1)[0]


def timed(fn, torch, reps=1):
    import numpy as np
    ts = []
    out = None
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return out, float(np.median(ts))


# ====================================================================== scal
def cmd_scal(spec, torch):
    """Time/peak memory/OOM for LR sizes `lr_sizes` (images are bicubically resized to size x size: only the cost is
    measured, no quality), T in Ts, one fixed `bsize` (the process is started once per batch size)."""
    import numpy as np
    import torch.nn.functional as F
    model = W.apply_mode(W.load_model(spec, torch), spec)
    ds = W.make_dataset(spec)
    n = min(int(spec["n_images"]), len(ds))
    scale = float(spec["scale"])
    bsize, reps = int(spec["bsize"]), int(spec.get("reps", 2))
    cap = int(spec.get("max_queries", 17_000_000))
    records = []
    for c in spec["lr_sizes"]:
        H = Wd = round(c * scale)
        if H * Wd > cap:
            records.append(dict(lr=c, n_queries=H * Wd, skipped=True))
            continue
        coord = W._make_coord((H, Wd), torch).cuda().unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / H
        cell[..., 1] *= 2 / Wd
        for T in spec["Ts"]:
            W.set_T(model, T)
            oom = False
            for i in range(n):
                inp0, _ = W.get_image(ds, i)
                inp = F.interpolate(inp0["inp"], size=(c, c), mode="bicubic", align_corners=False)
                inputs = dict(inp=inp, coord=coord, cell=cell)
                try:
                    if i == 0:
                        torch.manual_seed(0)
                        W.batched_predict(model, inputs, bsize, torch)       # warm-up / capture for these shapes
                    st = W.sharing_stats(inputs, bsize, torch)
                    times = []
                    peak = 0.0
                    for r in range(reps):
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        t0 = time.perf_counter()
                        torch.manual_seed(r + 31 * i)
                        W.batched_predict(model, inputs, bsize, torch)
                        torch.cuda.synchronize()
                        times.append(time.perf_counter() - t0)
                        peak = max(peak, torch.cuda.max_memory_allocated() / 2 ** 20)
                    rec = dict(lr=c, T=T, image=i, bsize=bsize, scale=scale, time_s=float(np.median(times)),
                               peak_mb=peak, oom=False)
                    rec.update(st)
                    records.append(rec)
                except Exception as e:
                    if not W.is_oom(e, torch):
                        raise
                    torch.cuda.empty_cache()
                    records.append(dict(lr=c, T=T, image=i, bsize=bsize, scale=scale, n_queries=H * Wd, oom=True))
                    oom = True
                    break
            if oom:
                continue
    return {"records": records}


# ====================================================================== noise_equiv
def cmd_noise_equiv(spec, torch):
    """Same seed, one model.query call on a block of full HR rows. Dumps (npz) per T:
         tok [N,C], fourier [N,dim,T]   input / output of the recurrent generator (N = Q for the original, U for the shared)
         feat_q [Q,C], cell_q [Q,2]     per-query inputs of the predictor
         phase [Q, T*out]                phase head output (only the original implementation returns it)
         recon [Q,3]                     reconstructed colours (normalised space)
       spec.replay=true (accelerated engine only): sp.fast_noise is replaced by sp.replay_noise, so the engine uses exactly the
       noise tensors the original draws from the same seed."""
    import numpy as np
    model = W.apply_mode(W.load_model(spec, torch), spec)
    ds = W.make_dataset(spec)
    inputs, targets = W.get_image(ds, int(spec.get("image", 0)))
    sl, rows, Wd = central_block(inputs, int(spec["n_q"]))
    c, ce = inputs["coord"][:, sl], inputs["cell"][:, sl]
    model.gen_feat(inputs["inp"])
    pred = model.predictor
    net = pred.layer
    cap = {}
    rec0 = net.recurrent

    def rec(*a, **k):
        out = rec0(*a, **k)
        tok = k["token"] if "token" in k else a[0]
        cap["tok"] = tok.detach()[..., 0].cpu().numpy()
        cap["fourier"] = out.detach().cpu().numpy()
        return out
    net.recurrent = rec
    fwd0 = pred.forward

    def fwd(*a, **k):
        cap["feat_q"] = a[0].detach().cpu().numpy()
        cap["cell_q"] = a[1].detach().cpu().numpy()
        out = fwd0(*a, **k)
        if isinstance(out, dict) and "phase" in out:
            cap["phase"] = out["phase"].detach().cpu().numpy()
        return out
    pred.forward = fwd

    replay = bool(spec.get("replay", False))
    curT = [0]
    if replay:
        import models.speculative as sp
        sp.fast_noise = lambda net_, T_, B=1, generator=None: sp.replay_noise(net_, curT[0])
    dim = pred.dim
    Wi, bi = pred.phase_input.weight[:, :, 0].detach(), pred.phase_input.bias.detach()
    Wp, bp = pred.phase.weight[:, :, 0].detach(), pred.phase.bias.detach()
    os.makedirs(spec["dump_dir"], exist_ok=True)
    meta = []
    for T in spec["Ts"]:
        W.set_T(model, T)
        curT[0] = T
        with torch.no_grad():
            torch.manual_seed(7)
            model.query(c, ce)                                        # warm-up (compilation)
            cap.clear()
            torch.manual_seed(int(spec.get("seed", 0)))
            recon = model.query(c, ce)["recon"]
        arrays = dict(cap)
        arrays["recon"] = recon[0].cpu().numpy()
        arrays.update(Wi=Wi.cpu().numpy(), bi=bi.cpu().numpy(), Wp=Wp.cpu().numpy(), bp=bp.cpu().numpy(), dim=np.array(dim))
        fn = os.path.join(spec["dump_dir"], f"{spec['label']}_T{T}.npz")
        np.savez_compressed(fn, **arrays)
        meta.append(dict(T=T, file=fn, n_queries=int(c.shape[1]), rows=rows, width=Wd))
    return {"records": meta}


# ====================================================================== arrange_exact
def cmd_arrange_exact(spec, torch):
    import numpy as np
    model = W.apply_mode(W.load_model(spec, torch), spec)
    ds = W.make_dataset(spec)
    inputs, _ = W.get_image(ds, int(spec.get("image", 0)))
    sl, rows, Wd = central_block(inputs, int(spec["n_q"]))
    c, ce = inputs["coord"][:, sl], inputs["cell"][:, sl]
    n = c.shape[1]
    model.gen_feat(inputs["inp"])
    lr_hw = inputs["inp"].shape[-2:]
    pid = pixel_ids(torch, lr_hw, c[0])
    pid_np = pid.cpu().numpy()
    rng = np.random.default_rng(0)
    # rank of every query inside its LR pixel -> ordering that puts pixel-mates into different batches
    order = np.argsort(pid_np, kind="stable")
    sid = pid_np[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
    gstart = np.repeat(starts, np.diff(np.r_[starts, len(order)]))
    ranks = np.empty(n, dtype=np.int64)
    ranks[order] = np.arange(n) - gstart
    split_perm = np.lexsort((pid_np, ranks))
    small = int(spec.get("small_chunk", 2000))
    rr, cc = np.divmod(np.arange(n), Wd)
    rc, xc = rows / 2, Wd / 2
    rect = np.flatnonzero((rr >= rows * 0.2) & (rr < rows * 0.7) & (cc >= Wd * 0.3) & (cc < Wd * 0.65))
    disk = np.flatnonzero((rr - rc) ** 2 + (cc - xc) ** 2 <= (0.3 * min(rows, Wd)) ** 2)
    rand10 = np.sort(rng.choice(n, max(1, n // 10), replace=False))
    arange = np.arange(n)
    variants = {
        "grid_1call": (arange, n),
        "grid_chunks": (arange, small),
        "shuffle_1call": (rng.permutation(n), n),
        "shuffle_chunks": (rng.permutation(n), small),
        "split_pixel_mates": (split_perm, small),
        "region_rect": (rect, small),
        "region_disk": (disk, small),
        "random_10pct": (rand10, small),
        "random_10pct_shuffled": (rng.permutation(rand10), small),
    }
    seed = int(spec.get("seed", 0))

    def run(perm, chunk):
        outs = []
        pt = torch.as_tensor(perm, device=c.device)
        with torch.no_grad():
            for l in range(0, len(perm), chunk):
                sel = pt[l:l + chunk]
                torch.manual_seed(seed)                    # same noise in every call
                outs.append(model.query(c[:, sel], ce[:, sel])["recon"][0].cpu())
        return torch.cat(outs).numpy()

    records = []
    for T in spec["Ts"]:
        W.set_T(model, T)
        # warm-up at both chunk shapes: with CUDA graphs the first call at a new shape is the capture call,
        # which draws its noise differently from the later replays, so it must not be the reference
        run(arange[:min(n, small)], small)
        run(arange, n)
        ref = run(arange, n)
        with torch.no_grad():
            torch.manual_seed(seed + 1)
            other = model.query(c, ce)["recon"][0].cpu().numpy()
        noise_scale = float(np.abs(ref - other).max())     # effect of a different noise draw, for context
        for name, (perm, chunk) in variants.items():
            if len(perm) == 0:                             # e.g. region outside a 1-row block (quick preset)
                records.append(dict(T=T, variant=name, n_queries=0, chunk=int(chunk), skipped="empty region"))
                continue
            run(perm, chunk)                               # warm-up: the last chunk may be a new shape (capture call)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = run(perm, chunk)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            d = np.abs(out - ref[perm])
            nroll = rollouts_for_order(pid[torch.as_tensor(perm, device=pid.device)], chunk, torch)
            records.append(dict(T=T, variant=name, n_queries=int(len(perm)), chunk=int(chunk), max_abs=float(d.max()),
                                mean_abs=float(d.mean()), rel_fro=float(np.linalg.norm(out - ref[perm]) / np.linalg.norm(ref[perm])),
                                rollouts=int(nroll), sharing_ratio=float(len(perm) / nroll), time_s=dt,
                                max_abs_other_seed=noise_scale))
    return {"records": records}


# ====================================================================== arrange_stat
def cmd_arrange_stat(spec, torch):
    """Full images, several (non-integer) scales and chunk sizes, three query orders; fresh noise per call."""
    import numpy as np
    import utils
    from functools import partial
    from lpips import LPIPS
    model = W.apply_mode(W.load_model(spec, torch), spec)
    lpips_net = LPIPS(net="alex").cuda()
    records = []
    for scale in spec["scales"]:
        sp_ = dict(spec, scale=scale)
        ds = W.make_dataset(sp_)
        psnr_fn = partial(utils.calc_psnr, dataset="div2k", scale=int(round(float(scale))))
        n = min(int(spec["n_images"]), len(ds))
        for T in spec["Ts"]:
            W.set_T(model, T)
            for i in range(n):
                inputs, targets = W.get_image(ds, i)
                coord, cell = inputs["coord"], inputs["cell"]
                Q = coord.shape[1]
                pid = pixel_ids(torch, inputs["inp"].shape[-2:], coord[0])
                g = torch.Generator().manual_seed(i)
                shuffle = torch.randperm(Q, generator=g).to(coord.device)
                ids_np = pid.cpu().numpy()
                o = np.argsort(ids_np, kind="stable")
                sid = ids_np[o]
                st = np.r_[0, np.flatnonzero(np.diff(sid)) + 1]
                gs = np.repeat(st, np.diff(np.r_[st, len(o)]))
                rk = np.empty(Q, dtype=np.int64)
                rk[o] = np.arange(Q) - gs
                split = torch.as_tensor(np.lexsort((ids_np, rk)), device=coord.device)
                orders = {"grid": None, "shuffled": shuffle, "split_pixel_mates": split}
                for chunk in spec["chunks"]:
                    for oname, perm in orders.items():
                        def go():
                            model.gen_feat(inputs["inp"])
                            out = torch.empty(1, Q, 3)
                            with torch.no_grad():
                                for l in range(0, Q, chunk):
                                    r = min(l + chunk, Q)
                                    sel = slice(l, r) if perm is None else perm[l:r]
                                    pr = model.query(coord[:, sel], cell[:, sel])["recon"].cpu()
                                    if perm is None:
                                        out[:, l:r] = pr
                                    else:
                                        out[:, perm[l:r].cpu()] = pr
                            return out
                        try:
                            if i == 0:
                                go()                                    # warm-up
                            torch.cuda.reset_peak_memory_stats()
                            torch.manual_seed(i)
                            out, dt = timed(go, torch, 1)
                        except Exception as e:
                            if not W.is_oom(e, torch):
                                raise
                            torch.cuda.empty_cache()
                            records.append(dict(scale=scale, T=T, image=i, chunk=chunk, order=oname, oom=True))
                            continue
                        peak = torch.cuda.max_memory_allocated() / 2 ** 20
                        ps, lp = W._metrics(out, inputs, targets, utils, psnr_fn, lpips_net, torch)
                        pids = pid if perm is None else pid[perm]
                        nroll = rollouts_for_order(pids, chunk, torch)
                        records.append(dict(scale=scale, T=T, image=i, chunk=chunk, order=oname, psnr=ps, lpips=lp,
                                            time_s=dt, peak_mb=peak, n_queries=Q, rollouts=nroll, sharing_ratio=Q / nroll,
                                            oom=False))
    return {"records": records}


# ====================================================================== adapt_vis
def cmd_adapt_vis(spec, torch):
    """For each image: fixed T (all of Ts, plus a second noise draw at the largest T), the adaptive allocation of every
    budget set in spec['budget_sets'] with the DCT score, and the random control. Saves float16 images and budget maps."""
    import numpy as np
    import utils
    model = W.apply_mode(W.load_model(spec, torch), spec)
    ds = W.make_dataset(spec)
    n = min(int(spec["n_images"]), len(ds))
    bsize, tile, metric = int(spec["bsize"]), int(spec.get("tile", 8)), spec.get("metric", "dct")
    os.makedirs(spec["dump_dir"], exist_ok=True)
    records = []
    for i in range(n):
        inputs, targets = W.get_image(ds, i)
        H, Wd = grid_hw(inputs)
        gt = utils.denormalize(targets["gt_img"])[0].clamp(0, 1)
        arrays = dict(gt=gt.cpu().numpy().astype(np.float16))
        times = {}
        Tmax = int(max(spec["Ts"]))

        def fixed(T, seed):
            def go():
                torch.manual_seed(seed)
                return W._adaptive_predict(model, inputs, bsize, None, T, torch)
            go()
            return timed(go, torch, 1)

        for T in spec["Ts"]:
            out, dt = fixed(T, 1000 + i)
            arrays[f"fixed{T}"] = flat_to_img(out, inputs, targets, utils).cpu().numpy().astype(np.float16)
            times[f"fixed{T}"] = dt
        out, dt = fixed(Tmax, 2000 + i)                                # noise floor: second draw at the largest T
        arrays[f"fixed{Tmax}_seed2"] = flat_to_img(out, inputs, targets, utils).cpu().numpy().astype(np.float16)
        for bset in spec["budget_sets"]:
            tag = "-".join(str(b) for b in bset)
            for m in (metric, "random"):
                def go():
                    torch.manual_seed(1000 + i)
                    bf = W._budget_flat(inputs, m, bset, tile, i, torch)
                    return W._adaptive_predict(model, inputs, bsize, bf, None, torch), bf
                go()
                (out, bf), dt = timed(go, torch, 1)
                key = f"{m}_{tag}"
                arrays[key] = flat_to_img(out, inputs, targets, utils).cpu().numpy().astype(np.float16)
                arrays[f"budget_{key}"] = bf.view(H, Wd)[:gt.shape[-2], :gt.shape[-1]].cpu().numpy().astype(np.uint8)
                times[key] = dt
        ih, iw = inputs["inp"].shape[-2:]
        arrays["tile_grid"] = np.array([ih // tile, iw // tile, tile])
        arrays["scale_hr_per_lr"] = np.array(H / ih)
        fn = os.path.join(spec["dump_dir"], f"{spec['label']}_img{i}.npz")
        np.savez_compressed(fn, **arrays)
        records.append(dict(image=i, file=fn, times=times, H=H, W=Wd))
    return {"records": records}


# ====================================================================== main
def main():
    cmd, spec_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    torch, root = W.boot(spec)
    import suitelib
    fns = {"scal": cmd_scal, "noise_equiv": cmd_noise_equiv, "arrange_exact": cmd_arrange_exact,
           "arrange_stat": cmd_arrange_stat, "adapt_vis": cmd_adapt_vis}
    try:
        result = fns[cmd](spec, torch)
        result["ok"] = True
    except Exception as e:
        result = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        print(result["trace"], file=sys.stderr)
    result["meta"] = dict(spec={k: v for k, v in spec.items() if k != "out"}, env=suitelib.env_info(), git=suitelib.git_info(root),
                          ckpt_sha256=suitelib.sha256_file(os.path.expanduser(spec["ckpt"])) if spec.get("hash_ckpt", True) else None)
    suitelib.write_json(spec["out"], result)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
