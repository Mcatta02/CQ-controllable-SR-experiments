#!/usr/bin/env python
"""
Where does the time go at T=64? The stage probe says ~75% of the accelerated time is 'other'
(outside predictor / decoder). This prints the heaviest CUDA kernels and ops of one full image.

  python profile_other.py <results_dir>/stages/fast.spec.json 64      # accelerated
  python profile_other.py <results_dir>/stages/orig.spec.json 64      # original
Run it in the recurrent_lte environment on the GPU used for the suite.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker

import glob
if len(sys.argv) > 1 and sys.argv[1].endswith(".json"):
    spec_path = os.path.abspath(sys.argv[1])
else:                                   # no argument: use the newest accelerated stage spec of a previous run
    here = os.path.dirname(os.path.abspath(__file__))
    cands = sorted(glob.glob(os.path.join(here, "results", "*", "stages", "fast.spec.json")))
    if not cands:
        sys.exit("usage: python suite/profile_other.py <results_dir>/stages/fast.spec.json [T]   (no previous run found)")
    spec_path = cands[-1]
print("spec:", spec_path)
spec = json.load(open(spec_path))
T = int(sys.argv[-1]) if sys.argv[-1].isdigit() else 64
torch, root = worker.boot(spec)
model = worker.apply_mode(worker.load_model(spec, torch), spec)
ds = worker.make_dataset(spec)
inputs, _ = worker.get_image(ds, 0)
worker.set_T(model, T)
bs = int(spec["bsize"])
torch.manual_seed(0)
worker.batched_predict(model, inputs, bs, torch)            # warm-up (compile / graph capture)
torch.cuda.synchronize()
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    torch.manual_seed(0)
    worker.batched_predict(model, inputs, bs, torch)
    torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=25, max_name_column_width=70))

# where do aten::cat / aten::stack come from? (input shapes and the Python call site)
def _dt(e):
    return getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
ka = prof.key_averages(group_by_input_shape=True)
rows = sorted([e for e in ka if e.key in ("aten::cat", "aten::stack")], key=_dt, reverse=True)[:4]
print("\n==== cat/stack call sites (largest first) ====")
for e in rows:
    print(f"{e.key}  calls={e.count}  device_time={_dt(e) / 1e3:.1f} ms")
    print("  input shapes:", str(e.input_shapes)[:300])