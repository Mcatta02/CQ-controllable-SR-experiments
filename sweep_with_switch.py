import yaml, csv
import torch
import torch._dynamo
import models
import datasets
import utils
from types import SimpleNamespace
import test as test_mod
from lpips import LPIPS
import warnings
import contextlib
import io
from models.convs import set_pointwise_kind

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
GPU = '1'

# Crossover found empirically (bench_pointwise_compile.py / crossover_sweep.py):
# below ~25000, linear+eager wins; at/above it, conv+compile wins, consistently
# across T=16/32/64.
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

    print(f"[pointwise_kind={'conv' if use_conv_compile else 'linear'} "
          f"compiled={use_conv_compile}] (eval_bsize={eval_bsize})")

    return model, loader, config

def run_all():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    model, loader, config = load_model_and_data()
    test_mod.config = config  # do_test's psnr_fn setup reads the module-level `config`
    with contextlib.redirect_stdout(io.StringIO()):
        lpips_net = LPIPS(net='alex').cuda()
    results = []

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
    [4, 16],
    [4, 32],
    [4, 64],
    [8, 16],
    [8, 32],
    [8, 64],
    [16, 32],
    [16, 64],
    [32, 64],
    [4, 8, 16],
    [4, 8, 32],
    [4, 8, 64],
    [4, 16, 32],
    [4, 16, 64],
    [4, 32, 64],
    [8, 16, 32],
    [8, 16, 64],
    [8, 32, 64],
    [16, 32, 64],
    [4, 8, 16, 32],
    [4, 8, 16, 64],
    [4, 8, 32, 64],
    [4, 16, 32, 64],
    [8, 16, 32, 64],
    [4, 8, 16, 32, 64]
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

    with open('sweep_results_x3_faster.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'label', 'psnr', 'lpips', 'time'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    run_all()