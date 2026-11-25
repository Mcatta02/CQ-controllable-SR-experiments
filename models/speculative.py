"""
models/speculative.py -- parallel verification, Jacobi and speculative decoding
for the real RecurrentNetwork (models/transformers.py). Non-invasive: it only
READS the weights/modules of an existing RecurrentNetwork, no existing file is
edited, so net.recurrent() stays the reference path.

What matches the real code (read from the repo):
  * step 0 input = token_layer(feature token); step t>0 input =
    input_layer(output_{t-1}); then causal_conv.step (a plain linear map, so
    NO conv buffer is needed -- verify is trivially batchable over positions).
  * per layer (CausalTransformerBlock): pre-LN, single-head linear attention
    with feature map elu+1 and eps 1e-10, SPE applied to q,k BEFORE the feature
    map, residuals, norm2 + pointwise MLP.
  * state per layer: S [B,R,Cv] (= sum k^T v), Z [B,1,R] (= sum k).

IMPORTANT -- the real model is stochastic through SPE:
  * pos codes (qbar,kbar) are redrawn at pos==0 of every recurrent() call
    (one per layer), and
  * the SPE gate noise is redrawn on EVERY step of EVERY layer.
Both are shared across the batch and depend on the step index only. So a
"trajectory" is a deterministic function of (token, noise realization). All
functions here take an explicit `noise` (list of per-layer dicts) so that
verify, Jacobi and speculative runs see the SAME noise as the run they are
compared against:
  replay_noise(net, T)  -- consumes the global RNG in exactly the order
                           net.recurrent() would (use for parity vs recurrent)
  fast_noise(net, T)    -- one randn per layer (use for deployment; same
                           distribution, different numbers than replay)
"""
import torch


def _pe(layer):
    return layer.attn.pos_encoding


def _pw(m, x):
    """Apply a pointwise (kernel-1) layer to channels-last x [B,L,C], whether the
    layer is a Linear, a channels-last PointwiseConv, or a raw Conv1d."""
    if isinstance(m, torch.nn.Conv1d) and type(m).__name__ != 'PointwiseConv':
        return m(x.transpose(1, 2)).transpose(1, 2)
    return m(x)


def _pw_seq(seq, x):
    for m in seq:
        x = _pw(m, x)
    return x


# --------------------------------------------------------------------------
# noise realizations
# --------------------------------------------------------------------------
@torch.no_grad()
def replay_noise(net, T, B=1):
    """Draw exactly what net.recurrent(num_pred=T) draws from the global RNG,
    in the same order (step outer, layer inner; pos code at step 0 first)."""
    L = len(net.layers)
    codes, g = [None] * L, [[] for _ in range(L)]
    for t in range(T):
        for l, layer in enumerate(net.layers):
            pe = _pe(layer)
            enc = pe.spe_encoder
            if t == 0:
                codes[l] = enc((B, pe.max_len))
            Fd, R = enc.in_features, enc.num_realizations
            dev = layer.attn.q.weight.device
            g[l].append(torch.randn(pe.spe_filter.code_shape + (R,), device=dev) / (Fd * R) ** 0.25)
    return [{'qbar': codes[l][0][0], 'kbar': codes[l][1][0], 'g': torch.stack(g[l])} for l in range(L)]


@torch.no_grad()
def fast_noise(net, T, B=1, generator=None):
    out = []
    for layer in net.layers:
        pe = _pe(layer)
        enc = pe.spe_encoder
        Fd, R = enc.in_features, enc.num_realizations
        codes = enc((B, pe.max_len))
        dev = layer.attn.q.weight.device
        g = torch.randn((T,) + tuple(pe.spe_filter.code_shape) + (R,), device=dev, generator=generator) / (Fd * R) ** 0.25
        out.append({'qbar': codes[0][0], 'kbar': codes[1][0], 'g': g})
    return out


@torch.no_grad()
def pe_tables(net, noise):
    """Fold gate + noise into per-position q/k code tables [T,H,F,R]
    (batch-independent, so computed once per run)."""
    tabs = []
    for layer, nz in zip(net.layers, noise):
        flt = _pe(layer).spe_filter
        assert flt.gated, 'ungated SPE filter not supported'
        gate = torch.sigmoid(flt.gate[..., None])  # [H,F,1]
        T = nz['g'].shape[0]
        a, b = torch.sqrt(1. - gate), torch.sqrt(gate)
        tabs.append((a * nz['qbar'][:T] + b * nz['g'], a * nz['kbar'][:T] + b * nz['g']))
        assert tabs[-1][0].shape[1] == 1, 'only single-head SPE is supported'
    return tabs


# --------------------------------------------------------------------------
# state
# --------------------------------------------------------------------------
def init_states(net, B):
    dev = net.out_layer.weight.device
    dt = net.out_layer.weight.dtype
    st = []
    for layer in net.layers:
        R = _pe(layer).spe_encoder.num_realizations
        Cv = layer.attn.v.weight.shape[0]
        st.append((torch.zeros(B, R, Cv, device=dev, dtype=dt), torch.zeros(B, 1, R, device=dev, dtype=dt)))
    return st


def advance(states, caches, w):
    """Fold the first-m steps of each point (w[b,i]=1 for i<m_b) into the state."""
    new = []
    for (S, Z), (k, v) in zip(states, caches):
        kw = k * w[..., None].to(k.dtype)
        new.append((S + kw.transpose(1, 2) @ v, Z + kw.sum(1, keepdim=True)))
    return new


# --------------------------------------------------------------------------
# parallel verify
# --------------------------------------------------------------------------
def _embed(net, token, y_prev, pos, lat_tok=None, skip_token=False):
    lat_prev = _pw(net.input_layer, y_prev)                # [B,K,E]
    if skip_token:                                         # caller guarantees pos > 0 everywhere
        lat = lat_prev
    else:
        if lat_tok is None:
            lat_tok = _pw(net.token_layer, token.transpose(1, 2))   # [B,1,E]
        first = pos == 0
        if first.dim() == 1:
            first = first[None].expand(y_prev.shape[0], -1)
        lat = torch.where(first[..., None], lat_tok, lat_prev)
    return net.causal_conv.step(lat)


def layer_verify(layer, x, state, tab, pos):
    attn = layer.attn
    qt, kt = tab
    h = layer.norm1(x)
    q, k, v = _pw(attn.q, h), _pw(attn.k, h), _pw(attn.v, h)
    q4, k4 = q.unsqueeze(2), k.unsqueeze(2)                # [B,K,1,F]
    if pos.dim() == 1:                                     # same positions for all points
        q = torch.einsum('bkhf,khfr->bkhr', q4, qt[pos])[:, :, 0]
        k = torch.einsum('bkhf,khfr->bkhr', k4, kt[pos])[:, :, 0]
    else:                                                  # per-point positions [B,K]
        q = torch.einsum('bkhf,bkhfr->bkhr', q4, qt[pos])[:, :, 0]
        k = torch.einsum('bkhf,bkhfr->bkhr', k4, kt[pos])[:, :, 0]
    q, k = attn.feature_fn(q), attn.feature_fn(k)
    S0, Z0 = state
    A = torch.tril(q @ k.transpose(1, 2))                  # [B,K,K]
    num = q @ S0 + A @ v
    den = (q * Z0).sum(-1) + A.sum(-1) + 1e-10             # [B,K]
    x = num / den.unsqueeze(-1) + x
    x = _pw_seq(layer.conv, layer.norm2(x)) + x
    return x, (k, v)


def net_verify(net, token, y_prev, pos, states, tables, lat_tok=None, skip_token=False):
    """token [B,in_dim,1] (as passed to recurrent); y_prev [B,K,C] = the input
    'previous output' of each step in the window (ignored where pos==0);
    pos [K] or [B,K]; returns outputs [B,K,C] and per-layer (k,v) caches."""
    x = _embed(net, token, y_prev, pos, lat_tok, skip_token)
    caches = []
    for layer, st, tab in zip(net.layers, states, tables):
        x, c = layer_verify(layer, x, st, tab, pos)
        caches.append(c)
    return _pw(net.out_layer, x), caches


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def _out_dim(net):
    return net.out_layer.weight.shape[0]


@torch.no_grad()
def generate_jacobi(net, token, T, tables, init=None, max_iters=None, callback=None):
    """Fixed-point iteration over the FULL length T. Exact after <= T iterations.
    Activation memory grows with T (use generate_speculative for O(K))."""
    B, C = token.shape[0], _out_dim(net)
    dev, dt = token.device, token.dtype
    Y = torch.zeros(B, T, C, device=dev, dtype=dt) if init is None else init.clone()
    pos = torch.arange(T, device=dev)
    zero = torch.zeros(B, 1, C, device=dev, dtype=dt)
    for it in range(1, (max_iters or T) + 1):
        y_prev = torch.cat([zero, Y[:, :-1]], 1)
        Ynew, _ = net_verify(net, token, y_prev, pos, init_states(net, B), tables)
        delta = (Ynew - Y).abs().amax((1, 2))
        Y = Ynew
        if callback is not None:
            callback(it, Y, delta)
    return Y


def draft_zeros(p, prev, K, T, G):
    return torch.zeros(prev.shape[0], K, prev.shape[-1], device=prev.device, dtype=prev.dtype)


def draft_repeat_last(p, prev, K, T, G):
    return prev[:, None].expand(-1, K, -1).clone()


def draft_carry(p, prev, K, T, G):
    """Windowed Jacobi: the draft is the verified-but-uncommitted outputs of
    earlier rounds (no draft model, no training)."""
    idx = (p[:, None] + torch.arange(K, device=p.device)).clamp(max=T - 1)
    return torch.gather(G, 1, idx[..., None].expand(-1, -1, G.shape[-1]))


@torch.no_grad()
def generate_speculative(net, token, T, tables, draft_fn=draft_carry, K=8, tol=1e-3):
    """Per-point speculative decoding with exact parallel verification.
    Memory is O(K) in the window, independent of T. `tol` is a float or a [T]
    tensor (per-position threshold). tol small enough to reject everything
    reproduces the sequential trajectory; tol>0 can drift (measure it)."""
    B, C = token.shape[0], _out_dim(net)
    dev, dt = token.device, token.dtype
    p = torch.zeros(B, dtype=torch.long, device=dev)
    prev = torch.zeros(B, C, device=dev, dtype=dt)
    states = init_states(net, B)
    Y = torch.zeros(B, T, C, device=dev, dtype=dt)
    G = torch.zeros(B, T, C, device=dev, dtype=dt)
    arK = torch.arange(K, device=dev)
    tol_t = tol if torch.is_tensor(tol) else None
    rounds = committed = active_pts = 0
    while bool((p < T).any()):
        active = p < T
        draft = draft_fn(p, prev, K, T, G)
        y_prev = torch.cat([prev[:, None], draft[:, :K - 1]], 1)
        idx = p[:, None] + arK
        pos = idx.clamp(max=T - 1)
        yv, caches = net_verify(net, token, y_prev, pos, states, tables)
        thr = tol_t[pos] if tol_t is not None else tol
        ok = (yv - draft).abs().amax(-1) <= thr
        n = ok.long().cumprod(1).sum(1)
        m = torch.minimum(torch.minimum(n + 1, torch.full_like(n, K)), T - p)
        m = torch.where(active, m, torch.zeros_like(m))
        w = arK[None] < m[:, None]
        bi, ki = w.nonzero(as_tuple=True)
        Y[bi, idx[bi, ki]] = yv[bi, ki]
        gv = (idx < T) & active[:, None]                   # carry ALL verified outputs forward
        gb, gk = gv.nonzero(as_tuple=True)
        G[gb, idx[gb, gk]] = yv[gb, gk]
        last = yv[torch.arange(B, device=dev), (m - 1).clamp(min=0)]
        prev = torch.where(active[:, None], last, prev)
        states = advance(states, caches, w)
        p = p + m
        rounds += 1
        committed += int(m.sum())
        active_pts += int(active.sum())
    return Y, {'rounds': rounds, 'mean_commit': committed / max(active_pts, 1)}