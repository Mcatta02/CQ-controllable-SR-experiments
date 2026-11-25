"""
Ablation variants of the post-processing fast path. Nothing in the repositories is edited.

install_post_variant(model, share, fuse, compile_post) is fast_post.install_fast_post with two
independent switches:
  share : run the recurrent generator once per distinct LR pixel (torch.unique(fid)) and gather
  fuse  : precompute M = Wp @ Wi and evaluate Eq. (4) of the paper (no Q x Cf x T tensor)
          instead of the two pointwise layers of Eq. (3)
The unfused path uses exactly the arithmetic of Eq. (3) with the same weights (it materialises
h = f + Wi[u;c] + bi with shape Q x Cf x T, then applies Wp), so share-only / fuse-only / both
differ only in the switched component.
"""
import types

import einops
import torch
from utils import fourier_decoding


def install_post_variant(model, share=True, fuse=True, compile_post=True):
    pred = model.predictor
    assert hasattr(pred, "phase_input") and hasattr(pred, "phase") and pred.layer.n_condition is None
    dim = pred.dim
    Wi, bi = pred.phase_input.weight[:, :, 0], pred.phase_input.bias
    Wp, bp = pred.phase.weight[:, :, 0], pred.phase.bias
    Wiu, Wic = Wi[:, :dim].contiguous(), Wi[:, dim:].contiguous()
    Wf = (Wp @ Wiu).contiguous()
    Wc = (Wp @ Wic).contiguous()
    cvec = (Wp @ bi + bp).contiguous()
    WpT = Wp.t().contiguous()

    def post_fused(fourier, feat, cell, coord, inv=None):
        if inv is not None:
            fourier = fourier[inv]
        const = feat @ WpT + cell @ Wc.t() + cvec
        phase = torch.einsum("qdt,od->qot", fourier, Wf) + const.unsqueeze(-1)
        coef, freq = pred.unreshape(fourier)
        phase = einops.rearrange(phase, "... c l -> ... (l c)")
        return fourier_decoding(coef, freq, phase, coord)

    def post_unfused(fourier, feat, cell, coord, inv=None):
        if inv is not None:
            fourier = fourier[inv]
        h = torch.einsum("qdt,cd->qct", fourier, Wiu) + (feat + cell @ Wic.t() + bi).unsqueeze(-1)   # Q x Cf x T
        phase = torch.einsum("qct,oc->qot", h, Wp) + bp.view(1, -1, 1)
        coef, freq = pred.unreshape(fourier)
        phase = einops.rearrange(phase, "... c l -> ... (l c)")
        return fourier_decoding(coef, freq, phase, coord)

    post = post_fused if fuse else post_unfused
    post_c = torch.compile(post, dynamic=True) if compile_post else post
    orig_forward = pred.forward

    def forward(self, feat, cell, coord, fid=None, *args):
        if self.num_pred == 0:
            return orig_forward(feat, cell, coord, *args)
        inv = None
        tok = feat
        if share and fid is not None:
            u, inv = torch.unique(fid, return_inverse=True)
            first = torch.empty(u.numel(), dtype=torch.long, device=fid.device)
            first.scatter_(0, inv, torch.arange(inv.numel(), device=fid.device))
            tok = feat[first]
        fourier = self.layer.recurrent(token=tok.unsqueeze(-1), num_pred=self.num_pred)
        return {"decoded": post_c(fourier, feat, cell, coord, inv)}

    pred.forward = types.MethodType(forward, pred)
    return model
