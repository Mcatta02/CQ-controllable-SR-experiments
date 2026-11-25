import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import os, torch
from torch.profiler import profile, ProfilerActivity, record_function
import parity_verify as pv
from models.fast_recurrent import install

T, BS, N = 64, int(os.environ.get("BS", 100000)), 100000
model, inputs = pv.load()
inp, coord, cell = (inputs[k].cuda() for k in ("inp", "coord", "cell"))
n = min(N, coord.shape[1])
model.predictor.num_pred = model.num_pred = model.num_preds = T

def run():
    for ql in range(0, n, BS):
        qr = min(ql + BS, n)
        model.query(coord[:, ql:qr], cell[:, ql:qr])

with torch.no_grad():
    model.gen_feat(inp)
    install(model, batch_size=BS)
    run(); run(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        run(); torch.cuda.synchronize()

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25))
prof.export_chrome_trace("query_trace.json")