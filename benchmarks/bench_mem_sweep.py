import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import os, sys, csv, time, math, yaml
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
import torch
import models, datasets, utils
import test as testmod
from types import SimpleNamespace

CONFIG_PATH = 'configs/test_one.yaml'
CHECKPOINT = './save/recurrent_lte_paper_repro/epoch-best.pth'
BATCH_SIZES = [5000, 10000, 25000, 50000, 100000]
NUM_PREDS = [4, 8, 16, 32, 64]                 # num_pred fissi da misurare
RUN_ADAPTIVE = False
N_IMAGES = 5                      # immagini misurate per cella
N_WARMUP = 1                      # immagini scartate (compilazione/cache)
OUT_CSV = 'mem_time_sweep3.csv'
MB = 2 ** 20

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

with open(CONFIG_PATH) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
if config.get('data_norm'):
    utils.set_normalizer(**config['data_norm'])
testmod.config = config

ds = datasets.make(config['test_dataset']['dataset'])
ds = datasets.make(config['test_dataset']['wrapper'], args={'dataset': ds})
loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=4, pin_memory=True)
samples = []
for k, (inp, tgt) in enumerate(loader):
    if k >= N_WARMUP + N_IMAGES:
        break
    samples.append(inp)

model = models.make(torch.load(CHECKPOINT, weights_only=False)['model'], load_sd=True).cuda().eval()
torch.cuda.synchronize()
param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / MB
base_mb = torch.cuda.memory_allocated() / MB
n_params = sum(p.numel() for p in model.parameters())
print(f'parametri: {n_params/1e6:.2f} M | pesi: {param_mb:.0f} MB | allocata dopo load: {base_mb:.0f} MB')


def run_one(inp, bs, adaptive, budgets, num_pred):
    inputs = {k: v.cuda() for k, v in inp.items()}
    if num_pred is not None:
        model.predictor.num_pred = num_pred
        model.num_pred = num_pred
        model.num_preds = num_pred
    testmod.args = SimpleNamespace(adaptive=adaptive, tile_size=64, budgets=budgets,
                                   metric='sobel', save_img=False, save_fourier=False)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    with torch.no_grad():
        testmod.batched_predict(model, inputs, bs, adaptive=adaptive,
                                tile_size=64, budgets=budgets, metric='sobel')
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    peak = torch.cuda.max_memory_allocated() / MB
    reserved = torch.cuda.max_memory_reserved() / MB
    del inputs
    return dt, peak, reserved


configs = [('fixed', f'num_pred={t}', False, None, t) for t in NUM_PREDS]
if RUN_ADAPTIVE:
    configs.append(('adaptive', f'budgets={ADAPTIVE_BUDGETS}', True, ADAPTIVE_BUDGETS, None))

rows = []
for kind, label, adaptive, budgets, T in configs:
    for bs in BATCH_SIZES:
        try:
            times, peaks, resv = [], [], []
            for i, inp in enumerate(samples):
                dt, peak, rs = run_one(inp, bs, adaptive, budgets, T)
                if i >= N_WARMUP:
                    times.append(dt); peaks.append(peak); resv.append(rs)
            row = dict(type=kind, label=label, batch_size=bs,
                       time_s=sum(times) / len(times),
                       peak_alloc_mb=max(peaks), peak_reserved_mb=max(resv),
                       peak_minus_weights_mb=max(peaks) - base_mb, status='ok')
        except torch.cuda.OutOfMemoryError:
            row = dict(type=kind, label=label, batch_size=bs, time_s=float('nan'),
                       peak_alloc_mb=float('nan'), peak_reserved_mb=float('nan'),
                       peak_minus_weights_mb=float('nan'), status='OOM')
        torch.cuda.empty_cache()
        rows.append(row)
        print(row)

with open(OUT_CSV, 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
print('salvato', OUT_CSV)