#!/usr/bin/env python
"""
Worker process. One call = one implementation (one root folder) = one JSON result.
    python suite/worker.py <eval|memory|stages|spec> <spec.json>

It chdirs into the implementation's root and puts it first on sys.path, so `models`, `datasets`
and `utils` are that implementation's own. Nothing in either repo is edited.

spec.json keys (all commands): root, ckpt, data_root, out, kind (linear|conv), mode, no_compile,
  scale, bsize.    mode: none | graph | post | full   (none = no fast path installed)
"""
import json
import math
import os
import sys
import time
import traceback


def boot(spec):
    root = os.path.abspath(os.path.expanduser(spec["root"]))
    here = os.path.dirname(os.path.abspath(__file__))
    os.chdir(root)
    sys.path.insert(0, root)
    sys.path.append(here)
    os.environ["POINTWISE_KIND"] = spec.get("kind", "linear")
    import torch
    if spec.get("no_compile"):
        # GPUs with compute capability < 7.0 (e.g. GTX 1080) cannot use Triton/Inductor:
        # make torch.compile a no-op so the rest of the fast path (graphs, sharing, fusion) still runs.
        def _identity_compile(fn=None, **kw):
            return (lambda f: f) if fn is None else fn
        torch.compile = _identity_compile
    return torch, root


def load_model(spec, torch):
    import models
    if spec.get("model_yaml"):
        # reference model with random weights (cost does not depend on the weights), e.g. configs/train-div2k/lte.yaml
        import yaml
        with open(os.path.expanduser(spec["model_yaml"])) as f:
            return models.make(yaml.safe_load(f)["model"]).cuda().eval()
    ck = torch.load(os.path.expanduser(spec["ckpt"]), map_location="cpu")
    return models.make(ck["model"], load_sd=True).cuda().eval()


def apply_flags(model, spec, flags):
    """Ablation switches (letters in `flags`):
         S sharing of rollouts per distinct LR pixel     F fused phase head (Eq. 4)
         E fast recurrent engine (pre-built SPE tables, token embedded once, padded batches)
         C torch.compile (engine step and post-processing; spec['no_compile'] must be set to match)
         G CUDA graph of the whole rollout (only with E)
    The compile switch is the spec key no_compile (boot() makes torch.compile a no-op)."""
    flags = set(flags)
    assert flags <= set("SFECG"), flags
    assert not ("G" in flags and "E" not in flags), "G needs E"
    assert ("C" in flags) == (not spec.get("no_compile", False)) or not flags & set("EC"), \
        "flag C and spec['no_compile'] disagree"
    if "E" in flags:
        os.environ["FAST_RECURRENT_CUDAGRAPH"] = "1" if "G" in flags else "0"
        from models.fast_recurrent import install
        install(model, batch_size=spec.get("bsize"))
    if flags & set("SF"):
        from fast_variants import install_post_variant
        install_post_variant(model, share="S" in flags, fuse="F" in flags, compile_post="C" in flags)
    return model


def apply_mode(model, spec):
    if spec.get("flags") is not None:
        return apply_flags(model, spec, spec["flags"])
    mode = spec.get("mode", "none")
    bs = spec.get("bsize")
    if mode in ("graph", "full"):
        from models.fast_recurrent import install
        install(model, batch_size=bs)
    if mode in ("post", "full"):
        from models.fast_post import install_fast_post
        try:
            install_fast_post(model, compile_post=not spec.get("no_compile", False))
        except TypeError:
            install_fast_post(model)
    return model


def set_T(model, T):
    model.predictor.num_pred = T
    model.num_pred = T
    model.num_preds = T


def make_dataset(spec):
    import datasets
    import utils
    utils.set_normalizer([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ds = datasets.make({"name": "image-folder",
                        "args": {"root_path": os.path.expanduser(spec["data_root"]),
                                 "first_k": int(spec["n_images"])}})
    s = float(spec["scale"])
    ds = datasets.make({"name": "sr-implicit-downsampled", "args": {"scale_min": s, "scale_max": s}},
                       args={"dataset": ds})
    return ds


def get_image(ds, i):
    inputs, targets = ds[i]
    return ({k: v.unsqueeze(0).cuda() for k, v in inputs.items()},
            {k: v.unsqueeze(0).cuda() for k, v in targets.items()})


COPY_ALL = False     # spec["copy_all"]=true restores the authors' harness (every output tensor is copied to the host)


def batched_predict(model, inputs, batch_size, torch):
    """Chunked reconstruction. By default only the reconstructed colours ("recon") are copied to the host,
    which is the result of super-resolution; the authors' test.py copies every output of the model
    (spec["copy_all"]=true), which adds a large, method-independent device-to-host cost to both
    implementations. Both implementations always go through the same function."""
    with torch.no_grad():
        inp, coord, cell = inputs["inp"], inputs["coord"], inputs["cell"]
        model.gen_feat(inp)
        n = coord.shape[1]
        ql = 0
        preds = {}
        while ql < n:
            qr = min(ql + batch_size, n)
            pred = model.query(coord[:, ql:qr], cell[:, ql:qr])
            for k, v in (pred.items() if COPY_ALL else [(k, pred[k]) for k in ("recon",) if k in pred]):
                preds.setdefault(k, [])
                preds[k].append([x.cpu() for x in v] if isinstance(v, list) else v.cpu())
            ql = qr
        for k, v in preds.items():
            if isinstance(v[0], list):
                for i in range(len(v[0])):
                    preds[k] = [torch.cat([x[i] for x in v], dim=1)]
            else:
                preds[k] = torch.cat(v, dim=1)
    return preds


def sharing_stats(inputs, bsize, torch):
    """Rollout-sharing statistics of one image, independent of the implementation.
    n_unique_global    : distinct nearest-LR pixels over the whole image
    n_rollouts_chunked : sum over chunks of `bsize` queries of the distinct pixels in each chunk
                         (= number of rollouts the shared implementation really runs)
    Query order and chunking are those of batched_predict."""
    coord = inputs["coord"][0]                       # [Q, 2] = (y, x) in [-1, 1]
    ih, iw = inputs["inp"].shape[-2:]
    iy = ((coord[:, 0] + 1) * ih / 2).floor().long().clamp_(0, ih - 1)
    ix = ((coord[:, 1] + 1) * iw / 2).floor().long().clamp_(0, iw - 1)
    idx = iy * iw + ix
    Q = idx.numel()
    glob = int(torch.unique(idx).numel())
    chunked = sum(int(torch.unique(idx[l:l + bsize]).numel()) for l in range(0, Q, bsize))
    return dict(n_queries=Q, n_lr_pixels=int(ih * iw), n_unique_global=glob, n_rollouts_chunked=chunked,
                share_ratio_global=Q / glob, share_ratio_chunked=Q / chunked)


def is_oom(e, torch):
    return isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e).lower()


# ====================================================================== eval
def cmd_eval(spec, torch):
    import numpy as np
    import utils
    from functools import partial
    from lpips import LPIPS

    model = apply_mode(load_model(spec, torch), spec)
    ds = make_dataset(spec)
    n = min(int(spec["n_images"]), len(ds))
    scale = float(spec["scale"])
    psnr_fn = partial(utils.calc_psnr, dataset="div2k", scale=int(round(scale)))
    lpips_net = LPIPS(net="alex").cuda()
    bsize = int(spec["bsize"])
    reps = int(spec.get("reps", 1))
    seeds = spec.get("seeds", [0])
    dump_dir, n_dump, crop = spec.get("dump_dir"), int(spec.get("n_dump", 0)), int(spec.get("dump_crop", 128))
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
    records = []
    share_cache = {}
    for T in spec["Ts"]:
        set_T(model, T)
        # cold pass: first call at this T (compilation / graph capture for the fast path)
        inputs, targets = get_image(ds, 0)
        torch.manual_seed(12345)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        try:
            batched_predict(model, inputs, bsize, torch)
        except Exception as e:
            if not is_oom(e, torch):
                raise
            torch.cuda.empty_cache()
            records.append(dict(T=T, image=0, seed=-1, oom=True))
            continue
        torch.cuda.synchronize(); cold_s = time.perf_counter() - t0
        records.append(dict(T=T, image=0, seed=-1, cold_s=cold_s, oom=False))
        oom = False
        for i in range(n):
            inputs, targets = get_image(ds, i)
            if i not in share_cache:
                share_cache[i] = sharing_stats(inputs, bsize, torch)
            for seed in seeds:
                times = []
                for rep in range(reps):
                    torch.manual_seed(1000 * int(seed) + i)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    try:
                        preds = batched_predict(model, inputs, bsize, torch)
                    except Exception as e:
                        if not is_oom(e, torch):
                            raise
                        torch.cuda.empty_cache(); oom = True
                        break
                    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
                if oom:
                    records.append(dict(T=T, image=i, seed=seed, oom=True))
                    break
                peak_mb = torch.cuda.max_memory_allocated() / 2 ** 20
                # ---- metrics exactly as in do_test (not timed)
                recon = preds["recon"].cuda()
                ih, iw = inputs["inp"].shape[-2:]
                s = math.sqrt(inputs["coord"].shape[1] / (ih * iw))
                recon = recon.view(1, round(ih * s), round(iw * s), -1).permute(0, 3, 1, 2).contiguous()
                recon = recon[..., :targets["gt_img"].shape[-2], :targets["gt_img"].shape[-1]]
                recon = utils.denormalize(recon).clamp_(0, 1)
                recon_f = recon.clone()
                for j in range(recon.shape[0]):
                    recon[j] = utils.discretize(recon[j])
                recon = utils.normalize(recon)
                lp = lpips_net(recon, targets["gt_img"]).mean().item()
                recon = utils.denormalize(recon).clamp_(0, 1)
                gt = utils.denormalize(targets["gt_img"])
                ps = float(psnr_fn(recon, gt))
                records.append(dict(T=T, image=i, seed=seed, psnr=ps, lpips=lp,
                                    time_s=float(np.median(times)), time_all=[float(t) for t in times],
                                    peak_mb=peak_mb, oom=False, **share_cache[i]))
                if dump_dir and i < n_dump:
                    H, W = recon_f.shape[-2:]
                    c = min(crop, H, W)
                    y0, x0 = (H - c) // 2, (W - c) // 2
                    np.savez_compressed(os.path.join(dump_dir, f"{spec['label']}_T{T}_s{seed}_i{i}.npz"),
                                        crop=recon_f[0, :, y0:y0 + c, x0:x0 + c].cpu().numpy(),
                                        gt=utils.denormalize(targets["gt_img"])[0, :, y0:y0 + c, x0:x0 + c].cpu().numpy())
            if oom:
                break
    return {"records": records}


# ====================================================================== memory
def cmd_memory(spec, torch):
    model = apply_mode(load_model(spec, torch), spec)
    ds = make_dataset(spec)
    inputs, _ = get_image(ds, 0)
    inp, coord, cell = inputs["inp"], inputs["coord"], inputs["cell"]
    n = min(int(spec["n_points"]), coord.shape[1])
    bs = int(spec["bsize"])
    reps = int(spec.get("reps", 2))
    model.gen_feat(inp)
    torch.cuda.synchronize()
    base_mb = torch.cuda.memory_allocated() / 2 ** 20

    def run():
        for ql in range(0, n, bs):
            model.query(coord[:, ql:min(ql + bs, n)], cell[:, ql:min(ql + bs, n)])

    records = []
    for T in spec["Ts"]:
        set_T(model, T)
        try:
            torch.manual_seed(0)
            run()                                       # warm-up (compile / capture)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            for _ in range(reps):
                run()
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / reps
            records.append(dict(T=T, bsize=bs, ms_per_100k=dt * 1000 / n * 1e5,
                                peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                                base_mb=base_mb, n_points=n, oom=False))
        except Exception as e:
            if not is_oom(e, torch):
                raise
            torch.cuda.empty_cache()
            records.append(dict(T=T, bsize=bs, oom=True, base_mb=base_mb, n_points=n))
    return {"records": records}


# ====================================================================== adaptive budgets
def _metrics(recon_flat, inputs, targets, utils, psnr_fn, lpips_net, torch):
    """PSNR / LPIPS exactly as in cmd_eval, from the flat [1, Q, 3] reconstruction."""
    recon = recon_flat.cuda()
    ih, iw = inputs["inp"].shape[-2:]
    s = math.sqrt(inputs["coord"].shape[1] / (ih * iw))
    recon = recon.view(1, round(ih * s), round(iw * s), -1).permute(0, 3, 1, 2).contiguous()
    recon = recon[..., :targets["gt_img"].shape[-2], :targets["gt_img"].shape[-1]]
    recon = utils.denormalize(recon).clamp_(0, 1)
    for j in range(recon.shape[0]):
        recon[j] = utils.discretize(recon[j])
    recon = utils.normalize(recon)
    lp = lpips_net(recon, targets["gt_img"]).mean().item()
    recon = utils.denormalize(recon).clamp_(0, 1)
    gt = utils.denormalize(targets["gt_img"])
    return float(psnr_fn(recon, gt)), lp


def _budget_flat(inputs, metric, budgets, tile, seed, torch):
    """Per-query budget (flat, in coord order). metric: sobel | variance | dct | random.
    Equal-mass quantile buckets (difficulty.difficulty_to_budget), so every metric, including the
    random control, has the same average number of steps."""
    try:
        from difficulty import compute_tile_difficulty, difficulty_to_budget, budget_map_to_coord_order
    except ImportError:                                   # fall back to the copy next to this file
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        from difficulty import compute_tile_difficulty, difficulty_to_budget, budget_map_to_coord_order
    inp = inputs["inp"]
    ih, iw = inp.shape[-2:]
    s = math.sqrt(inputs["coord"].shape[1] / (ih * iw))
    H, W = round(ih * s), round(iw * s)
    if metric == "random":
        g = torch.Generator(device=inp.device)
        g.manual_seed(int(seed))
        diff = torch.rand(1, ih // tile, iw // tile, device=inp.device, generator=g)
    else:
        diff = compute_tile_difficulty(inp, tile_size=tile, metric=metric)
    return budget_map_to_coord_order(difficulty_to_budget(diff, tuple(budgets)), H, W)[0]


def _adaptive_predict(model, inputs, bsize, budget_flat, T_fixed, torch):
    """Queries are grouped by their budget; every group runs through the normal (shared) pipeline with
    its own T in chunks of `bsize`. Groups are tile-aligned, so all queries of an LR pixel are in the
    same group and the rollout sharing is kept. budget_flat=None -> fixed T for all queries."""
    with torch.no_grad():
        coord, cell = inputs["coord"], inputs["cell"]
        model.gen_feat(inputs["inp"])
        n = coord.shape[1]
        if budget_flat is None:
            groups = [(int(T_fixed), None)]
        else:
            groups = [(int(b), (budget_flat == b).nonzero(as_tuple=True)[0]) for b in torch.unique(budget_flat).tolist()]
        out = None
        for T, idx in groups:
            set_T(model, T)
            m = n if idx is None else idx.numel()
            for l in range(0, m, bsize):
                r = min(l + bsize, m)
                if idx is None:
                    pred = model.query(coord[:, l:r], cell[:, l:r])["recon"].cpu()
                    sel = slice(l, r)
                else:
                    sel_g = idx[l:r]
                    pred = model.query(coord[:, sel_g], cell[:, sel_g])["recon"].cpu()
                    sel = sel_g.cpu()
                if out is None:
                    out = torch.empty(1, n, pred.shape[-1])
                out[:, sel] = pred
        return out


def cmd_adaptive(spec, torch):
    """Fixed T = max budget (e.g. 64) versus patch-wise budgets (e.g. {32, 64}): tiles that are easy by a
    difficulty measure get the smaller budget. Also records fixed T at the mean budget (matched average T).
    spec: budgets (list), metrics (e.g. [sobel, variance, dct]; 'random' is allowed as a control), tile, n_images, reps.
    Timing = difficulty estimation + allocation + grouping + reconstruction + copy to host (steady state:
    one untimed pass per image and method first, so that no graph capture is timed)."""
    import numpy as np
    import utils
    from functools import partial
    from lpips import LPIPS

    model = apply_mode(load_model(spec, torch), spec)
    ds = make_dataset(spec)
    n = min(int(spec["n_images"]), len(ds))
    psnr_fn = partial(utils.calc_psnr, dataset="div2k", scale=int(round(float(spec["scale"]))))
    lpips_net = LPIPS(net="alex").cuda()
    bsize, reps, tile = int(spec["bsize"]), int(spec.get("reps", 1)), int(spec.get("tile", 8))
    budgets = list(spec["budgets"])
    T_avg = int(round(sum(budgets) / len(budgets)))
    T_max = int(max(budgets))
    # reference 1: fixed T = largest budget (what one would run without budgets);
    # reference 2: fixed T = mean of the budgets (same average number of steps as the mixed allocation)
    methods = [("fixed_max", None, T_max)] + ([("fixed", None, T_avg)] if T_avg != T_max else []) \
        + [(f"fixed_T{t}", None, int(t)) for t in spec.get("fixed_Ts", [])] \
        + [(m, m, None) for m in spec["metrics"]]
    records = []
    for i in range(n):
        inputs, targets = get_image(ds, i)
        for name, metric, Tf in methods:
            def once():
                torch.manual_seed(1000 * 0 + i)
                torch.cuda.synchronize(); t0 = time.perf_counter()
                bf = None if metric is None else _budget_flat(inputs, metric, budgets, tile, i, torch)
                torch.cuda.synchronize(); t_est = time.perf_counter() - t0
                rec = _adaptive_predict(model, inputs, bsize, bf, Tf, torch)
                torch.cuda.synchronize()
                return rec, time.perf_counter() - t0, t_est, bf
            try:
                once()                                                        # warm-up (graph capture)
                ts, tes = [], []
                for _ in range(reps):
                    rec, t, te, bf = once()
                    ts.append(t); tes.append(te)
            except Exception as e:
                if not is_oom(e, torch):
                    raise
                torch.cuda.empty_cache()
                records.append(dict(method=name, image=i, oom=True))
                continue
            ps, lp = _metrics(rec, inputs, targets, utils, psnr_fn, lpips_net, torch)
            mean_T = float(Tf) if bf is None else float(bf.float().mean())
            records.append(dict(method=name, budgets=budgets, T_max=T_max, image=i, psnr=ps, lpips=lp,
                                time_s=float(np.median(ts)), t_est_s=float(np.median(tes)), mean_T=mean_T, oom=False))
    return {"records": records}


# ====================================================================== resolution (time/memory only)
def _make_coord(shape, torch):
    seqs = []
    for n in shape:
        r = 1.0 / n
        seqs.append(-1 + r + 2 * r * torch.arange(n).float())
    return torch.stack(torch.meshgrid(*seqs, indexing="ij"), -1).view(-1, 2)


def cmd_resolution(spec, torch):
    """Time and peak memory against image size. The first image is centre-cropped to LR size c x c
    (c in spec['lr_sizes']; limited by the image) and reconstructed at spec['scale']. No metrics."""
    import numpy as np
    model = apply_mode(load_model(spec, torch), spec)
    ds = make_dataset(spec)
    inputs0, _ = get_image(ds, 0)
    scale = float(spec["scale"])
    bsize, reps = int(spec["bsize"]), int(spec.get("reps", 3))
    records = []
    for c in spec["lr_sizes"]:
        inp = inputs0["inp"]
        ih, iw = inp.shape[-2:]
        ch, cw = min(c, ih), min(c, iw)
        y0, x0 = (ih - ch) // 2, (iw - cw) // 2
        H, W = round(ch * scale), round(cw * scale)
        coord = _make_coord((H, W), torch).cuda().unsqueeze(0)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / H
        cell[..., 1] *= 2 / W
        inputs = dict(inp=inp[..., y0:y0 + ch, x0:x0 + cw].contiguous(), coord=coord, cell=cell)
        for T in spec["Ts"]:
            set_T(model, T)
            try:
                torch.manual_seed(0)
                batched_predict(model, inputs, bsize, torch)           # warm-up / capture
                times = []
                for r in range(reps):
                    torch.manual_seed(r)
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    batched_predict(model, inputs, bsize, torch)
                    torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
                rec = dict(T=T, lr_h=ch, lr_w=cw, time_s=float(np.median(times)),
                           peak_mb=torch.cuda.max_memory_allocated() / 2 ** 20, oom=False)
                rec.update(sharing_stats(inputs, bsize, torch))      # includes n_queries
                records.append(rec)
            except Exception as e:
                if not is_oom(e, torch):
                    raise
                torch.cuda.empty_cache()
                records.append(dict(T=T, lr_h=ch, lr_w=cw, n_queries=H * W, oom=True))
    return {"records": records}


# ====================================================================== stages
class Probe:
    def __init__(self, torch):
        self.torch = torch
        self.reset()

    def reset(self):
        self.ev, self.cnt, self.rows = {}, {}, {}

    def wrap(self, obj, name, label, rows_of=None):
        f = getattr(obj, name)
        torch = self.torch

        def w(*a, **k):
            s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            s.record()
            r = f(*a, **k)
            e.record()
            self.ev.setdefault(label, []).append((s, e))
            self.cnt[label] = self.cnt.get(label, 0) + 1
            if rows_of is not None:
                try:
                    self.rows[label] = self.rows.get(label, 0) + rows_of(a, k)
                except Exception:
                    pass
            return r

        setattr(obj, name, w)

    def ms(self, label):
        self.torch.cuda.synchronize()
        return sum(s.elapsed_time(e) for s, e in self.ev.get(label, []))


def cmd_stages(spec, torch):
    model = apply_mode(load_model(spec, torch), spec)
    ds = make_dataset(spec)
    bsize = int(spec["bsize"])
    reps = int(spec.get("reps", 2))
    p = Probe(torch)
    p.wrap(model, "gen_feat", "gen_feat")
    if hasattr(model, "_fid"):
        p.wrap(model, "_fid", "fid")
    pred = model.predictor
    p.wrap(pred, "forward", "predictor", lambda a, k: a[0].shape[0])
    net = getattr(pred, "layer", None)
    if net is not None and hasattr(net, "recurrent"):
        p.wrap(net, "recurrent", "recurrent", lambda a, k: (k["token"] if "token" in k else a[0]).shape[0])
    p.wrap(model.decoder, "forward", "decoder")
    inputs, _ = get_image(ds, 0)
    records = []
    for T in spec["Ts"]:
        set_T(model, T)
        torch.manual_seed(0)
        batched_predict(model, inputs, bsize, torch)       # warm
        p.reset()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for r in range(reps):
            torch.manual_seed(r)
            batched_predict(model, inputs, bsize, torch)
        torch.cuda.synchronize()
        total = (time.perf_counter() - t0) / reps * 1000
        g = {k: p.ms(k) / reps for k in ("gen_feat", "fid", "predictor", "recurrent", "decoder")}
        other = total - g["gen_feat"] - g["fid"] - g["predictor"] - g["decoder"]
        records.append(dict(T=T, total_ms=total, other_ms=other,
                            rows_predictor=p.rows.get("predictor", 0) / reps,
                            rows_recurrent=p.rows.get("recurrent", 0) / reps, **{k + "_ms": v for k, v in g.items()}))
    return {"records": records}


# ====================================================================== spec (negative result)
def _graphed(fn, torch):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    torch.cuda.synchronize()
    return g.replay, out


def _timeit(fn, torch, n=5):
    fn(); torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def cmd_spec(spec, torch):
    import models.speculative as sp
    model = load_model(spec, torch)
    ds = make_dataset(spec)
    inputs, _ = get_image(ds, 0)
    inp, coord, cell = inputs["inp"], inputs["coord"], inputs["cell"]
    model.gen_feat(inp)
    T = int(spec.get("T", 64))
    set_T(model, T)
    net = model.predictor.layer
    N = int(spec.get("n_tokens", 10000))
    idx = torch.linspace(0, coord.shape[1] - 1, N, device=coord.device).long()
    cap, orig = {}, net.recurrent

    def wrapped(*a, **k):
        cap["token"] = k["token"].detach()
        return orig(*a, **k)

    net.recurrent = wrapped
    try:
        with torch.no_grad():
            model.query(coord[:, idx], cell[:, idx])
    finally:
        net.recurrent = orig
    token = cap["token"]
    B = token.shape[0]
    out = {"T": T, "n_tokens": B}
    with torch.no_grad():
        # ---- exact parallel verification vs. the sequential recurrence (same noise)
        torch.manual_seed(0)
        ref = net.recurrent(token=token, num_pred=T).transpose(1, 2).contiguous()
        scale = float(ref.abs().max())
        torch.manual_seed(0)
        tables = sp.pe_tables(net, sp.replay_noise(net, T))
        y_prev = torch.cat([torch.zeros_like(ref[:, :1]), ref[:, :-1]], 1)
        pos = torch.arange(T, device=ref.device)
        yv, _ = sp.net_verify(net, token, y_prev, pos, sp.init_states(net, B), tables)
        out["verify_max_abs"] = float((yv - ref).abs().max())
        out["output_scale"] = scale
        # ---- seed-to-seed spread of the recurrence itself (noise floor)
        torch.manual_seed(1)
        other = net.recurrent(token=token, num_pred=T).transpose(1, 2).contiguous()
        out["seed_to_seed_max_abs"] = float((ref - other).abs().max())
        # ---- Jacobi iteration from zeros: error vs. iteration
        curve = []

        def cb(it, Y, delta):
            e = (Y - ref).abs()
            curve.append(dict(iter=it, max_err=float(e.max()), mean_err=float(e.mean())))

        sp.generate_jacobi(net, token, T, tables, max_iters=T, callback=cb)
        out["jacobi"] = curve
        # ---- cost of one verify+commit round vs. sequential steps (eager and graphed)
        seq_eager = _timeit(lambda: net.recurrent(token=token, num_pred=T), torch)
        try:
            rep, _ = _graphed(lambda: net.recurrent(token=token, num_pred=T), torch)
            seq_graph = _timeit(rep, torch)
        except Exception as e:
            seq_graph = None
            out["graph_error"] = str(e)[:200]
        out["seq_eager_ms"] = seq_eager
        out["seq_graph_ms"] = seq_graph
        full_pass = _timeit(lambda: sp.net_verify(net, token, y_prev, pos, sp.init_states(net, B), tables), torch)
        out["full_verify_pass_ms"] = full_pass
        rounds = []
        C = sp._out_dim(net)
        for K in spec.get("Ks", [2, 4, 8, 16]):
            yp = torch.randn(B, K, C, device=token.device) * 0.3
            ps = (10 + torch.arange(K, device=token.device)).expand(B, K).contiguous()
            w = torch.ones(B, K, device=token.device)
            st = sp.init_states(net, B)

            def rnd():
                y, caches = sp.net_verify(net, token, yp, ps, st, tables)
                return y, sp.advance(st, caches, w)

            te = _timeit(rnd, torch)
            tg = None
            try:
                r2, _ = _graphed(rnd, torch)
                tg = _timeit(r2, torch)
            except Exception:
                pass
            rounds.append(dict(K=K, eager_ms=te, graph_ms=tg))
        out["rounds"] = rounds
    return out


# ====================================================================== main
def main():
    cmd, spec_path = sys.argv[1], sys.argv[2]
    with open(spec_path) as f:
        spec = json.load(f)
    torch, root = boot(spec)
    global COPY_ALL
    COPY_ALL = bool(spec.get("copy_all", False))
    import suitelib
    fns = {"eval": cmd_eval, "memory": cmd_memory, "stages": cmd_stages, "spec": cmd_spec, "resolution": cmd_resolution, "adaptive": cmd_adaptive}
    try:
        result = fns[cmd](spec, torch)
        result["ok"] = True
    except Exception as e:
        result = {"ok": False, "error": repr(e), "trace": traceback.format_exc()[-3000:]}
        print(result["trace"], file=sys.stderr)
    result["meta"] = dict(spec={k: v for k, v in spec.items() if k not in ("out",)},
                          env=suitelib.env_info(), git=suitelib.git_info(root),
                          ckpt_sha256=suitelib.sha256_file(os.path.expanduser(spec["ckpt"])) if spec.get("hash_ckpt", True) else None)
    suitelib.write_json(spec["out"], result)
    sys.exit(0 if result["ok"] else 1)


if __name__ == "__main__":
    main()
