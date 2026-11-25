"""
Parallel verification + Jacobi + speculative drivers for a linear-attention
autoregressive regressor with output feedback (RecurrentNetwork-style).

Contents
--------
ToyLinearCausalAttention : step() (sequential), verify() (parallel window),
                           advance() (commit a per-point number of steps)
ToyRecurrentNet          : stack of the above with output feedback
generate_sequential      : reference, one step at a time
generate_jacobi          : fixed-point iteration, full-T parallel verify
generate_speculative     : windowed draft->verify->commit with per-point
                           pointers (memory O(K), independent of T)

The toy net mirrors the *structure* of models/transformers.py (input layer on
the previous output, position codes per step, causal linear attention state
S, Z, output layer). The attention/feature-map details (elu+1, eps, heads,
CausalConv1d buffer, SPE codes) must be matched to the real repo when wiring
this in -- see the TODO list at the bottom of this file.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def phi(x):
    return F.elu(x) + 1.0


class ToyLinearCausalAttention(nn.Module):
    def __init__(self, dim, heads, eps=1e-6):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.h, self.dh, self.eps = dim, heads, dim // heads, eps
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)

    def init_state(self, B, device=None, dtype=None):
        S = torch.zeros(B, self.h, self.dh, self.dh, device=device, dtype=dtype)
        Z = torch.zeros(B, self.h, self.dh, device=device, dtype=dtype)
        return S, Z

    def _qkv(self, x):  # x: (B, K, D) -> each (B, H, K, dh)
        B, K, _ = x.shape

        def f(t):
            return t.view(B, K, self.h, self.dh).transpose(1, 2)

        return phi(f(self.q(x))), phi(f(self.k(x))), f(self.v(x))

    # ---- sequential (generate) mode: S_t = S_{t-1} + k_t^T v_t ----------
    def step(self, x, state):  # x: (B, D)
        B = x.shape[0]
        q, k, v = (t[:, :, 0] for t in self._qkv(x[:, None]))  # (B,H,dh)
        S, Z = state
        S = S + k.unsqueeze(-1) * v.unsqueeze(-2)
        Z = Z + k
        num = torch.einsum("bhd,bhde->bhe", q, S)
        den = (q * Z).sum(-1, keepdim=True) + self.eps
        return self.o((num / den).reshape(B, self.dim)), (S, Z)

    # ---- parallel (verify) mode over a window of K given inputs ---------
    # out_i = q_i S0 + sum_{j<=i} (q_i . k_j) v_j  -- no K copies of S needed
    def verify(self, x, state):  # x: (B, K, D)
        B, K, _ = x.shape
        q, k, v = self._qkv(x)
        S0, Z0 = state
        A = torch.einsum("bhid,bhjd->bhij", q, k).tril()
        num = torch.einsum("bhid,bhde->bhie", q, S0) + A @ v
        den = torch.einsum("bhid,bhd->bhi", q, Z0) + A.sum(-1) + self.eps
        out = (num / den.unsqueeze(-1)).transpose(1, 2).reshape(B, K, self.dim)
        return self.o(out), (k, v)

    # ---- commit: fold the first m_b steps of each point into the state --
    @staticmethod
    def advance(state, kv, weights):  # weights: (B, K) of 0/1
        S, Z = state
        k, v = kv
        kw = k * weights[:, None, :, None].to(k.dtype)
        S = S + torch.einsum("bhkd,bhke->bhde", kw, v)
        Z = Z + kw.sum(2)
        return S, Z


class ToyRecurrentNet(nn.Module):
    def __init__(self, out_dim=3, dim=16, heads=2, layers=2, max_len=64):
        super().__init__()
        self.out_dim, self.dim, self.max_len = out_dim, dim, max_len
        self.inp = nn.Linear(out_dim, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.attn = nn.ModuleList(ToyLinearCausalAttention(dim, heads) for _ in range(layers))
        self.mlp = nn.ModuleList(
            nn.Sequential(nn.Linear(dim, 2 * dim), nn.GELU(), nn.Linear(2 * dim, dim))
            for _ in range(layers)
        )
        self.out = nn.Linear(dim, out_dim)

    def init_states(self, B, device=None, dtype=None):
        return [a.init_state(B, device, dtype) for a in self.attn]

    def step(self, y_prev, pos, states):  # y_prev (B,out), pos (B,)
        x = self.inp(y_prev) + self.pos(pos)
        new = []
        for a, m, s in zip(self.attn, self.mlp, states):
            h, s = a.step(x, s)
            x = x + h
            x = x + m(x)
            new.append(s)
        return self.out(x), new

    def verify(self, y_prev_seq, pos, states):  # (B,K,out), (B,K)
        x = self.inp(y_prev_seq) + self.pos(pos)
        caches = []
        for a, m, s in zip(self.attn, self.mlp, states):
            h, kv = a.verify(x, s)
            x = x + h
            x = x + m(x)
            caches.append(kv)
        return self.out(x), caches

    def advance(self, states, caches, weights):
        return [ToyLinearCausalAttention.advance(s, kv, weights) for s, kv in zip(states, caches)]


# --------------------------------------------------------------------------
# Drivers
# --------------------------------------------------------------------------
def _ref(model):
    p = next(model.parameters())
    return p.device, p.dtype


@torch.no_grad()
def generate_sequential(model, B, T):
    dev, dt = _ref(model)
    y = torch.zeros(B, model.out_dim, device=dev, dtype=dt)
    states = model.init_states(B, dev, dt)
    ys = []
    for t in range(T):
        pos = torch.full((B,), t, device=dev, dtype=torch.long)
        y, states = model.step(y, pos, states)
        ys.append(y)
    return torch.stack(ys, 1)


@torch.no_grad()
def generate_jacobi(model, B, T, init=None, tol=1e-6, max_iters=None):
    """Fixed-point iteration Y <- f(shift(Y)). After i iterations the first i
    positions are exact, so it converges in <= T iterations (+1 to detect).
    NOTE: verifies the full length T at once -> activation memory grows with T
    (O(T^2) attention window). Use generate_speculative for O(K) memory."""
    dev, dt = _ref(model)
    Y = torch.zeros(B, T, model.out_dim, device=dev, dtype=dt) if init is None else init.clone()
    start = torch.zeros(B, 1, model.out_dim, device=dev, dtype=dt)
    pos = torch.arange(T, device=dev).expand(B, T)
    max_iters = max_iters or T + 1
    conv_iter = torch.full((B,), max_iters, device=dev, dtype=torch.long)
    history = []
    for it in range(1, max_iters + 1):
        inp = torch.cat([start, Y[:, :-1]], 1)
        Ynew, _ = model.verify(inp, pos, model.init_states(B, dev, dt))
        delta = (Ynew - Y).abs().amax(dim=(1, 2))
        history.append(delta.max().item())
        newly = (delta <= tol) & (conv_iter == max_iters)
        conv_iter = torch.where(newly, torch.full_like(conv_iter, it), conv_iter)
        Y = Ynew
        if (delta <= tol).all():
            break
    return Y, {"iters": it, "per_point_iters": conv_iter, "delta_history": history}


# draft_fn(p, prev, K, T) -> (B, K, out): guesses for positions p..p+K-1
def draft_zeros(p, prev, K, T):
    return torch.zeros(prev.shape[0], K, prev.shape[1], device=prev.device, dtype=prev.dtype)


def draft_repeat_last(p, prev, K, T):
    return prev[:, None].expand(-1, K, -1).clone()


def make_oracle_draft(Y_true, noise=0.0, noise_frac=0.0, generator=None):
    """Test helper: draft = true trajectory, optionally corrupted at a random
    fraction of positions by `noise`."""

    def fn(p, prev, K, T):
        B = prev.shape[0]
        idx = (p[:, None] + torch.arange(K, device=p.device)).clamp(max=T - 1)
        d = torch.gather(Y_true, 1, idx[..., None].expand(-1, -1, Y_true.shape[-1])).clone()
        if noise > 0:
            mask = torch.rand(B, K, generator=generator, device=d.device) < noise_frac
            d = d + mask[..., None] * noise * torch.randn(d.shape, generator=generator, device=d.device, dtype=d.dtype)
        return d

    return fn


@torch.no_grad()
def generate_speculative(model, B, T, draft_fn, K=8, tol=1e-3):
    """Per-point speculative decoding with exact (parallel) verification.

    Round, per point with pointer p (state built from inputs y_{<p}, `prev` =
    y_{p-1}):
      1. draft proposes y^_p .. y^_{p+K-1}
      2. verify inputs [prev, y^_p .. y^_{p+K-2}] in ONE parallel pass ->
         y'_p .. y'_{p+K-1}. y'_p is always exact.
      3. n = #leading draft steps with |y' - y^|_inf <= tol; commit
         m = min(n+1, K, T-p) outputs (the "+1" bonus is the verified output
         after the last accepted draft step).
      4. state advances over the m committed *inputs*; p += m.
    tol=0 (or drafts equal to the truth) => identical to sequential.
    tol>0 => outputs are y' but the state was built from drafted inputs, so
    error can drift; measure it (see test_spec_decode.py)."""
    dev, dt = _ref(model)
    p = torch.zeros(B, dtype=torch.long, device=dev)
    prev = torch.zeros(B, model.out_dim, device=dev, dtype=dt)
    states = model.init_states(B, dev, dt)
    Y = torch.zeros(B, T, model.out_dim, device=dev, dtype=dt)
    arK = torch.arange(K, device=dev)
    rounds, committed, active_pts = 0, 0, 0
    assert T <= model.max_len
    while (p < T).any():
        active = p < T
        draft = draft_fn(p, prev, K, T)
        inp = torch.cat([prev[:, None], draft[:, : K - 1]], 1)
        pos = (p[:, None] + arK).clamp(max=T - 1)
        yv, caches = model.verify(inp, pos, states)
        ok = (yv - draft).abs().amax(-1) <= tol  # (B,K)
        n = ok.long().cumprod(1).sum(1)
        m = torch.minimum(torch.minimum(n + 1, torch.full_like(n, K)), T - p)
        m = torch.where(active, m, torch.zeros_like(m))
        w = arK[None] < m[:, None]  # (B,K) committed steps
        bi, ki = w.nonzero(as_tuple=True)
        Y[bi, (p[:, None] + arK)[bi, ki]] = yv[bi, ki]
        last = yv[torch.arange(B, device=dev), (m - 1).clamp(min=0)]
        prev = torch.where(active[:, None], last, prev)
        states = model.advance(states, caches, w)
        p = p + m
        rounds += 1
        committed += int(m.sum())
        active_pts += int(active.sum())
    return Y, {"rounds": rounds, "mean_commit_per_active_point_round": committed / max(active_pts, 1)}


# --------------------------------------------------------------------------
# TODO when wiring into the real repo (needs models/attentions.py and
# models/transformers.py, which I have not seen in this conversation):
#  1. Match LinearCausalAttention exactly: feature map, eps/normaliser,
#     heads, any scaling, and whether S_t includes the current k_t^T v_t
#     (the plan says yes: q_t @ S_t).
#  2. CausalConv1d: verify() needs a window version that prepends the stored
#     last (kernel-1) per-layer inputs; advance() must also update that buffer
#     using only the committed steps (per-point gather).
#  3. SPE positional codes: gather per-point positions p+arange(K) from the
#     precomputed table (clamp to max_len-1; steps >= T are masked by m).
#  4. LayerNorm / residual / token_layer / out_layer are per-position, so they
#     batch over K for free -- keep them identical to the step path.
# --------------------------------------------------------------------------
