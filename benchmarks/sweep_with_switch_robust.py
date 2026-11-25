import os as _os, sys as _sys  # run from repo root: python benchmarks/<script>.py
_ROOT = _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))
_sys.path[:0] = [_ROOT, _os.path.join(_ROOT, 'tests')]

import yaml, csv
import torch
import models
import datasets
import utils
from types import SimpleNamespace
import test as test_mod  # reuse do_test, batched_predict
from lpips import LPIPS
import warnings
import contextlib
import io

# Both of these only exist in the OPTIMIZED code (models/convs.py,
# torch._dynamo). When the pre_optimized_baseline/*.py files are swapped
# into models/, neither is available -- fall back to plain eager/linear so
# the SAME script runs against either code version without edits.
try:
    from models.convs import set_pointwise_kind
    import torch._dynamo
    HAS_OPTIMIZATIONS = True
except ImportError:
    HAS_OPTIMIZATIONS = False

warnings.filterwarnings(
    "ignore",
    message="The parameter 'pretrained' is deprecated"
)

warnings.filterwarnings(
    "ignore",
    message="Arguments other than a weight enum or `None` for 'weights' are deprecated"
)

CONFIG_PATH = 'configs/test_one.yaml'
CHECKPOINT = './save/recurrent_lte_paper_repro/epoch-best.pth'
GPU = '3'

# Crossover found empirically (bench_pointwise_compile.py / crossover_sweep.py):
# below ~25000, linear+eager wins; at/above it, conv+compile wins, consistently
# across T=16/32/64. Only applied when HAS_OPTIMIZATIONS is True.
POINTWISE_CROSSOVER_BATCH = 25000

def load_model_and_data():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset = datasets.make(config['test_dataset']['dataset'])
    dataset = datasets.make(config['test_dataset']['wrapper'], args={'dataset': dataset})
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['test_dataset']['batch_size'],
                                          num_workers=16, pin_memory=True)

    use_conv_compile = False
    if HAS_OPTIMIZATIONS:
        # Pick pointwise kind + compile based on THIS run's actual eval batch size
        # (config['eval_bsize'], which is what gets passed to do_test below), per the
        # measured crossover. set_pointwise_kind() must be called BEFORE models.make() --
        # the kind is baked into which layer class gets constructed, not changeable after.
        eval_bsize = config.get('eval_bsize')
        use_conv_compile = eval_bsize is not None and eval_bsize >= POINTWISE_CROSSOVER_BATCH
        set_pointwise_kind('conv' if use_conv_compile else 'linear')

    model_spec = torch.load(CHECKPOINT, weights_only=False)['model']
    model = models.make(model_spec, load_sd=True).cuda()

    if use_conv_compile:
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        model.predictor = torch.compile(model.predictor, dynamic=True)

    label = ("conv+compiled" if use_conv_compile else
             ("linear+eager" if HAS_OPTIMIZATIONS else "pre-optimization baseline"))
    print(f"[{label}] (eval_bsize={config.get('eval_bsize')})")

    return model, loader, config

def run_all():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    model, loader, config = load_model_and_data()
    test_mod.config = config  # do_test's psnr_fn setup reads the module-level `config`
    with contextlib.redirect_stdout(io.StringIO()):
        lpips_net = LPIPS(net='alex').cuda()
    results = []

    # fixed baselines
    for np_ in [4, 8, 16, 32, 64]:
        model.predictor.num_pred = np_
        model.num_pred = np_
        model.num_preds = np_
        test_mod.args = SimpleNamespace(adaptive=False, tile_size=64, budgets=None,
                                         save_img=False, save_fourier=False)
        psnr, lpips, t = test_mod.do_test(model, loader, save_dir=None,
                                           batch_size=config.get('eval_bsize'), lpips_net=lpips_net)
        results.append({'type': 'fixed', 'label': f'num_pred={np_}', 'psnr': psnr, 'lpips': lpips, 'time': t})
        print(results[-1])
        torch.cuda.empty_cache()

    # adaptive sweeps — vary the budget set to trace out different points on the curve
    budget_sets = [
    [4, 8],
    [8, 16],
    [16, 32],
    [32, 64],

    [4, 8, 16],
    [8, 16, 32],
    [16, 32, 64]
    ]
    for metric in ['sobel']:  # can also be used with variance and dct
        for budgets in budget_sets:
            test_mod.args = SimpleNamespace(
                adaptive=True,
                tile_size=64,
                budgets=budgets,
                metric=metric,
                save_img=False,
                save_fourier=False
            )

            psnr, lpips, t = test_mod.do_test(
                model,
                loader,
                save_dir=None,
                batch_size=config.get('eval_bsize'),
                lpips_net=lpips_net
            )

            results.append({
                'type': f'adaptive-{metric}',
                'label': f'budgets={budgets}',
                'psnr': psnr,
                'lpips': lpips,
                'time': t
            })

            print(results[-1])
            torch.cuda.empty_cache()

    suffix = "optimized" if HAS_OPTIMIZATIONS else "baseline"
    out_path = f'sweep_results_{suffix}.csv'
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'label', 'psnr', 'lpips', 'time'])
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved {out_path}")

if __name__ == '__main__':
    run_all()
