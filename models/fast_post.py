import types
import einops
import torch
from utils import fourier_decoding

def install_fast_post(model, compile_post=True):
    pred = model.predictor
    assert hasattr(pred, "phase_input") and hasattr(pred, "phase") and pred.layer.n_condition is None
    dim = pred.dim
    Wi, bi = pred.phase_input.weight[:, :, 0], pred.phase_input.bias
    Wp, bp = pred.phase.weight[:, :, 0], pred.phase.bias
    Wf = (Wp @ Wi[:, :dim]).contiguous()
    Wc = (Wp @ Wi[:, dim:]).contiguous()
    cvec = (Wp @ bi + bp).contiguous()
    WpT = Wp.t().contiguous()

    def post(fourier, feat, cell, coord, inv=None):
        if inv is not None:
            fourier = fourier[inv]
        const = feat @ WpT + cell @ Wc.t() + cvec
        phase = torch.einsum("qdt,od->qot", fourier, Wf) + const.unsqueeze(-1)
        coef, freq = pred.unreshape(fourier)
        phase = einops.rearrange(phase, "... c l -> ... (l c)")
        return fourier_decoding(coef, freq, phase, coord)

    post_c = torch.compile(post, dynamic=True) if compile_post else post
    orig_forward = pred.forward

    def forward(self, feat, cell, coord, fid=None, *args):
        if self.num_pred == 0:
            return orig_forward(feat, cell, coord, *args)
        inv = None
        tok = feat
        if fid is not None:
            u, inv = torch.unique(fid, return_inverse=True)
            first = torch.empty(u.numel(), dtype=torch.long, device=fid.device)
            first.scatter_(0, inv, torch.arange(inv.numel(), device=fid.device))
            tok = feat[first]
        fourier = self.layer.recurrent(token=tok.unsqueeze(-1), num_pred=self.num_pred)
        return {"decoded": post_c(fourier, feat, cell, coord, inv)}

    pred.forward = types.MethodType(forward, pred)
    return model