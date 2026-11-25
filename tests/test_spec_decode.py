"""Run: python test_spec_decode.py   (or pytest test_spec_decode.py)
All checks in float64. Pattern follows parity_check.py: parallel path vs the
sequential path must agree to numerical noise."""
import torch
from spec_decode import (ToyRecurrentNet, generate_sequential, generate_jacobi,
                         generate_speculative, draft_zeros, draft_repeat_last,
                         make_oracle_draft)

torch.manual_seed(0)
B, T, OUT = 6, 64, 3
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def make_model():
    m = ToyRecurrentNet(out_dim=OUT, dim=16, heads=2, layers=2, max_len=T).to(DEV).double()
    for p in m.parameters():  # larger weights so the dynamics are non-trivial
        p.data *= 1.5
    return m.eval()


def test_verify_matches_step_teacher_forced():
    m = make_model()
    K, warm = 8, 5
    with torch.no_grad():
        states = m.init_states(B, DEV, torch.float64)
        y = torch.zeros(B, OUT, device=DEV, dtype=torch.float64)
        for t in range(warm):  # build a nonzero state
            y, states = m.step(y, torch.full((B,), t, device=DEV), states)
        inp = torch.randn(B, K, OUT, device=DEV, dtype=torch.float64)
        pos = (warm + torch.arange(K, device=DEV)).expand(B, K)
        yv, caches = m.verify(inp, pos, states)
        s, outs = states, []
        for i in range(K):
            o, s = m.step(inp[:, i], pos[:, i], s)
            outs.append(o)
        err = (yv - torch.stack(outs, 1)).abs().max().item()
        # advance over all K must equal stepping K times
        adv = m.advance(states, caches, torch.ones(B, K, device=DEV))
        serr = max(max((a[0] - b[0]).abs().max().item(), (a[1] - b[1]).abs().max().item())
                   for a, b in zip(adv, s))
    print(f"verify vs step max err: {err:.2e}, advance(all K) vs step state err: {serr:.2e}")
    assert err < 1e-10 and serr < 1e-10


def test_advance_partial_per_point():
    m = make_model()
    K = 8
    with torch.no_grad():
        states = m.init_states(B, DEV, torch.float64)
        inp = torch.randn(B, K, OUT, device=DEV, dtype=torch.float64)
        pos = torch.arange(K, device=DEV).expand(B, K)
        _, caches = m.verify(inp, pos, states)
        mm = torch.randint(0, K + 1, (B,), device=DEV)
        adv = m.advance(states, caches, (torch.arange(K, device=DEV)[None] < mm[:, None]))
        worst = 0.0
        for b in range(B):  # per-point reference: step exactly mm[b] times
            s = m.init_states(1, DEV, torch.float64)
            for i in range(int(mm[b])):
                _, s = m.step(inp[b:b + 1, i], pos[b:b + 1, i], s)
            for (Sa, Za), (Sb, Zb) in zip(adv, s):
                worst = max(worst, (Sa[b:b + 1] - Sb).abs().max().item(), (Za[b:b + 1] - Zb).abs().max().item())
    print(f"per-point partial advance max err: {worst:.2e}")
    assert worst < 1e-10


def test_jacobi_exact_and_bounded():
    m = make_model()
    ref = generate_sequential(m, B, T)
    Y, info = generate_jacobi(m, B, T, tol=1e-12)
    err = (Y - ref).abs().max().item()
    print(f"jacobi: err {err:.2e}, iters {info['iters']} (bound T+1={T+1}), "
          f"per-point iters min/mean/max "
          f"{info['per_point_iters'].min().item()}/{info['per_point_iters'].double().mean().item():.1f}/"
          f"{info['per_point_iters'].max().item()}")
    assert err < 1e-9 and info["iters"] <= T + 1


def test_speculative_exact_for_tight_tol():
    m = make_model()
    ref = generate_sequential(m, B, T)
    K = 8
    g = torch.Generator(device=DEV).manual_seed(1)
    cases = {
        "zeros": (draft_zeros, T),                       # nothing accepted -> 1 step/round
        "repeat_last": (draft_repeat_last, None),
        "oracle": (make_oracle_draft(ref), T // K),      # all accepted -> K steps/round
        "noisy_oracle": (make_oracle_draft(ref, noise=0.5, noise_frac=0.1, generator=g), None),
    }
    for name, (fn, expect_rounds) in cases.items():
        Y, st = generate_speculative(m, B, T, fn, K=K, tol=1e-9)
        err = (Y - ref).abs().max().item()
        print(f"spec[{name:12s}] err {err:.2e} rounds {st['rounds']:3d} "
              f"mean commit {st['mean_commit_per_active_point_round']:.2f}")
        assert err < 1e-8, name
        if expect_rounds is not None:
            assert st["rounds"] == expect_rounds, (name, st["rounds"])


def test_speculative_drift_report():
    """tol>0: not asserted, just measures the compounding error."""
    m = make_model()
    ref = generate_sequential(m, B, T)
    g = torch.Generator(device=DEV).manual_seed(2)
    fn = make_oracle_draft(ref, noise=0.02, noise_frac=1.0, generator=g)
    for tol in (1e-3, 1e-2, 5e-2):
        Y, st = generate_speculative(m, B, T, fn, K=8, tol=tol)
        print(f"drift tol={tol:g}: max err {(Y - ref).abs().max().item():.2e}, "
              f"rounds {st['rounds']}, mean commit {st['mean_commit_per_active_point_round']:.2f}")


if __name__ == "__main__":
    test_verify_matches_step_teacher_forced()
    test_advance_partial_per_point()
    test_jacobi_exact_and_bounded()
    test_speculative_exact_for_tight_tol()
    test_speculative_drift_report()
    print("ALL OK")
