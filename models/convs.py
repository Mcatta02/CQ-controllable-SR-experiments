import torch
from torch import nn
from torch.nn import functional as F


class CausalConv1d(nn.Conv1d):
    def __init__(self, mask_center, *args, **kwargs):
        super().__init__(*args, **kwargs)

        i, o, l = self.weight.shape
        mask = torch.zeros((i, o, l))
        mask.data[:, :, :l//2 + int(not mask_center)] = 1
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Masked weight is computed out of place. The previous `self.weight.data *= self.mask`
        # mutated a parameter inside forward on every call (extra kernel, and hostile to
        # torch.compile). Same function: masked taps contribute nothing and get zero gradient.
        return F.conv1d(x, self.weight * self.mask, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def step(self, x):
        """Single recurrent step. x: [batch, 1, in_channels] -> [batch, 1, out_channels].

        Equivalent to `self(F.pad(x.transpose(1, 2), (0, 1)))[:, :, -1]` for a kernel-3,
        padding-1 conv: the pad leaves only tap 0 touching the real input, so the whole
        conv collapses to a linear map with the (masked) tap-0 weights.
        """
        return F.linear(x, (self.weight * self.mask)[:, :, 0], self.bias)


class PointwiseLinear(nn.Linear):
    """nn.Linear replacement for Conv1d(kernel_size=1) that still loads old checkpoints.

    Conv1d weights are [out, in, 1]; Linear weights are [out, in]. The extra dim is squeezed
    on load, so existing checkpoints keep working. Operates on [..., in] (channels last).
    """

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + 'weight'
        w = state_dict.get(key)
        if w is not None and w.dim() == 3:
            state_dict[key] = w.squeeze(-1)
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)


class PointwiseConv(nn.Conv1d):
    """Conv1d(kernel_size=1), adapted to the same channels-last [B, L, in] calling
    convention as PointwiseLinear so callers don't need to know which one is active.

    This *is* an nn.Conv1d (not a wrapper around one), so its state_dict keys and weight
    shape ([out, in, 1]) match the original checkpoint layout exactly -- no conversion.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__(in_dim, out_dim, kernel_size=1)

    def forward(self, x):
        # x: [B, L, in] -> [B, in, L] -> conv -> [B, L, out]
        return super().forward(x.transpose(1, 2)).transpose(1, 2)


# Which implementation make_pointwise() returns by default. Conv1d(k=1) and Linear are the
# same op; which is faster depends on batch size (see bench_adaptive.py / microbenchmarks):
# Linear tends to win at small batches (~10000 and below), Conv1d at large batches (~100000+),
# because cuDNN's autotuning (cudnn.benchmark=True) gives Conv1d a per-shape tuned kernel that
# cuBLAS's addmm heuristic doesn't have an equivalent for. Measure on your own batch size
# before trusting either default.
#
# Override per-run with the POINTWISE_KIND env var, or call set_pointwise_kind(...) before
# building the model. Switching kind changes the traced graph, so under torch.compile it
# costs a fresh compile the first time a given kind is used -- expected, not a bug.
import os  # noqa: E402
_POINTWISE_KIND = os.environ.get('POINTWISE_KIND', 'linear').lower()
_POINTWISE_IMPLS = {'linear': PointwiseLinear, 'conv': PointwiseConv}


def make_pointwise(in_dim, out_dim, kind=None):
    kind = (kind or _POINTWISE_KIND).lower()
    if kind not in _POINTWISE_IMPLS:
        raise ValueError(f"unknown pointwise kind {kind!r}, expected one of {list(_POINTWISE_IMPLS)}")
    return _POINTWISE_IMPLS[kind](in_dim, out_dim)


def set_pointwise_kind(kind):
    """Change the process-wide default used by future make_pointwise(kind=None) calls.
    Call this before constructing the model -- it has no effect on layers already built."""
    global _POINTWISE_KIND
    if kind not in _POINTWISE_IMPLS:
        raise ValueError(f"unknown pointwise kind {kind!r}, expected one of {list(_POINTWISE_IMPLS)}")
    _POINTWISE_KIND = kind