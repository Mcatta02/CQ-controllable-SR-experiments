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
BATCH_SIZE = 100000  # forced, not read from config.eval_bsize -- conv wins at this size

# tile_size values to sweep -- edit this to whatever patch sizes you actually want tested
TILE_SIZES = [1, 2]

BUDGET_SETS = [
    [4, 8],
    [8, 16],
    [16, 32],
    [32, 64],
    [4, 8, 16],
    [8, 16, 32],
    [16, 32, 64],
]


def load_model_and_data():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset = datasets.make(config['test_dataset']['dataset'])
    dataset = datasets.make(config['test_dataset']['wrapper'], args={'dataset': dataset})
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['test_dataset']['batch_size'],
                                          num_workers=16, pin_memory=True)

    # Set the pointwise kind BEFORE the model is constructed. The env var is only read once,
    # at import time, inside models/convs.py -- setting it here (after `import models` already
    # ran above) would be too late. set_pointwise_kind() is a direct call, so it always applies
    # to whatever gets built next, regardless of import order.
    from models.convs import set_pointwise_kind
    set_pointwise_kind('conv')

    model_spec = torch.load(CHECKPOINT, weights_only=False)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    return model, loader, config


def run_all():
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU
    model, loader, config = load_model_and_data()
    test_mod.config = config  # do_test's psnr_fn setup reads the module-level `config`
    with contextlib.redirect_stdout(io.StringIO()):
        lpips_net = LPIPS(net='alex').cuda()
    results = []

    # adaptive only -- fixed baselines dropped, adaptive already known to dominate them
    for metric in ['sobel']:  # can also be used with variance and dct
        for tile_size in TILE_SIZES:
            for budgets in BUDGET_SETS:
                test_mod.args = SimpleNamespace(
                    adaptive=True,
                    tile_size=tile_size,
                    budgets=budgets,
                    metric=metric,
                    save_img=False,
                    save_fourier=False,
                )

                psnr, lpips, t = test_mod.do_test(
                    model,
                    loader,
                    save_dir=None,
                    batch_size=BATCH_SIZE,
                    lpips_net=lpips_net,
                )

                results.append({
                    'type': f'adaptive-{metric}',
                    'tile_size': tile_size,
                    'label': f'budgets={budgets}',
                    'psnr': psnr,
                    'lpips': lpips,
                    'time': t,
                })

                print(results[-1])
                torch.cuda.empty_cache()

    with open('sweep_results_tilesize_conv.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'tile_size', 'label', 'psnr', 'lpips', 'time'])
        writer.writeheader()
        writer.writerows(results)


if __name__ == '__main__':
    run_all()