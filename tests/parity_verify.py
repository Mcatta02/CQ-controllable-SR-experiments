"""
Parity + first real measurements for parallel verify / Jacobi / speculative.
Place models/speculative.py in the repo's models/ dir, this file at repo root.

    python parity_verify.py

  T0  noise sensitivity: how different are two recurrent() runs with different
      seeds? (the SPE gate noise is redrawn every step -> is the model really
      stochastic at inference? this decides how "deterministic" claims read)
  T1  PARITY: teacher-force recurrent()'s own outputs through ONE parallel
      verify pass, same noise (replayed from the same seed) -> must match.
  T2  Jacobi from zeros: error vs the sequential trajectory after n iterations.
  T3  speculative (carry-over draft, no draft model): rounds / mean commit /
      error vs tol.
  T4  timing: recurrent() vs one full-T verify pass vs speculative.
"""
import os as _os, sys as _sys  # run from repo root: python tests/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import os
import sys
import time

# respects an already-set CUDA_VISIBLE_DEVICES; defaults to 3 only if unset
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "3")

import yaml
import torch

import models
import datasets
import utils
import models.speculative as sp

CONFIG_PATH = "configs/test_one.yaml"
CHECKPOINT = "./save/recurrent_lte_paper_repro/epoch-best.pth"
N_POINTS = int(os.environ.get("N_POINTS", 10000))  # batch of query points (real eval batch size)
T = 64
SEED = 0
K = 8

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
def capture_token(model, inputs):
    """Run the real model once and grab the exact token recurrent() receives."""
    inp, coord, cell = inputs["inp"].cuda(), inputs["coord"].cuda(), inputs["cell"].cuda()
    model.gen_feat(inp)
    n = coord.shape[1]
    idx = torch.linspace(0, n - 1, N_POINTS, device=inp.device).long()
    net = model.predictor.layer
    cap, orig = {}, net.recurrent

    def wrapped(*a, **kw):
        cap["token"] = kw["token"].detach()
        return orig(*a, **kw)

    net.recurrent = wrapped
    try:
        model.query_t(coord[:, idx], cell[:, idx], T)
    finally:
        net.recurrent = orig
    return net, cap["token"]


def timeit(fn, n=3):
    fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / n * 1000


def main():
    model, inputs = load()
    net, token = capture_token(model, inputs)
    assert net.n_condition is None, "n_condition not supported"
    B = token.shape[0]
    print(f"net={type(net).__name__}, token {tuple(token.shape)}, T={T}")

    with torch.no_grad():
        # reference trajectory, noise from SEED
        torch.manual_seed(SEED)
        ref = net.recurrent(token=token, num_pred=T).transpose(1, 2).contiguous()   # [B,T,C]
        scale = ref.abs().max().item()

        # T0 noise sensitivity
        torch.manual_seed(SEED + 1)
        other = net.recurrent(token=token, num_pred=T).transpose(1, 2).contiguous()
        print(f"\n[T0] max|ref(seed0)-ref(seed1)| = {(ref - other).abs().max().item():.3e}  (ref max abs {scale:.3e})")

        # replayed noise == the noise ref was generated with
        torch.manual_seed(SEED)
        noise = sp.replay_noise(net, T)
        tables = sp.pe_tables(net, noise)

        # T1 parity: teacher-forced, one parallel pass
        y_prev = torch.cat([torch.zeros_like(ref[:, :1]), ref[:, :-1]], 1)
        pos = torch.arange(T, device=ref.device)
        yv, _ = sp.net_verify(net, token, y_prev, pos, sp.init_states(net, B), tables)
        d = (yv - ref).abs().max().item()
        print(f"[T1] verify(teacher-forced) vs recurrent: max abs {d:.3e} (rel {d / scale:.3e})  "
              f"{'OK' if d / scale < 1e-4 else 'CHECK'}")

        # T2 Jacobi from zeros
        report = {1, 2, 4, 8, 16, 32, 64}
        print("[T2] Jacobi from zeros -- error vs sequential trajectory:")

        def cb(it, Y, delta):
            if it in report:
                err = (Y - ref).abs()
                print(f"     iter {it:3d}: max err {err.max().item():.3e}  "
                      f"mean err {err.mean().item():.3e}  frac points err<1e-3*scale "
                      f"{(err.amax((1, 2)) < 1e-3 * scale).float().mean().item():.3f}")

        sp.generate_jacobi(net, token, T, tables, max_iters=T, callback=cb)

        # T3 speculative, carry-over draft
        print(f"[T3] speculative K={K}, carry-over draft (no draft model):")
        for tol in (1e-5, 1e-4, 1e-3, 1e-2):
            Y, st = sp.generate_speculative(net, token, T, tables, K=K, tol=tol * scale)
            e = (Y - ref).abs()
            print(f"     tol {tol:g}*scale: rounds {st['rounds']:3d} (seq {T}), mean commit {st['mean_commit']:.2f}, "
                  f"max err {e.max().item():.3e}, mean err {e.mean().item():.3e}")

    # T4 timing (noise pre-drawn; its cost reported separately)
    print("[T4] timing (ms):")
    t_rec = timeit(lambda: net.recurrent(token=token, num_pred=T))
    with torch.no_grad():
        pos = torch.arange(T, device=ref.device)
        t_ver = timeit(lambda: sp.net_verify(net, token, y_prev, pos, sp.init_states(net, B), tables))
        t_spe = timeit(lambda: sp.generate_speculative(net, token, T, tables, K=K, tol=1e-3 * scale))
        t_nz_fast = timeit(lambda: sp.fast_noise(net, T))
    print(f"     recurrent (T={T}):            {t_rec:8.1f}")
    print(f"     ONE full-T verify pass:      {t_ver:8.1f}   (Jacobi cost = iterations x this)")
    print(f"     speculative tol 1e-3*scale:  {t_spe:8.1f}")
    print(f"     fast_noise (one-off, per run): {t_nz_fast:6.1f}")
    print(f"     peak GPU memory this script: {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB "
          f"(device {os.environ['CUDA_VISIBLE_DEVICES']}, POINTWISE_KIND={os.environ.get('POINTWISE_KIND', 'linear')})")


if __name__ == "__main__":
    main()