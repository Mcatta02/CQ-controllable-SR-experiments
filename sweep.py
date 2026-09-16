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
GPU = '0'

def load_model_and_data():
    with open(CONFIG_PATH) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset = datasets.make(config['test_dataset']['dataset'])
    dataset = datasets.make(config['test_dataset']['wrapper'], args={'dataset': dataset})
    loader = torch.utils.data.DataLoader(dataset, batch_size=config['test_dataset']['batch_size'],
                                          num_workers=4, pin_memory=True)
    model_spec = torch.load(CHECKPOINT)['model']
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

    # fixed baselines
    for np_ in [4, 8, 16, 32, 64]:
        model.predictor.num_pred = np_
        model.num_pred = np_
        model.num_preds = np_
        test_mod.args = SimpleNamespace(adaptive=False, tile_size=8, budgets=None,
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

    #[4, 8, 16, 32],
    #[4, 8, 16, 64],
    #[4, 8, 32, 64],
    #[4, 16, 32, 64],
    #[8, 16, 32, 64],

    #[4, 8, 16, 32, 64]
    ]
    for metric in ['sobel', 'variance','dct']:
        for budgets in budget_sets:
            test_mod.args = SimpleNamespace(
                adaptive=True,
                tile_size=8,
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

    with open('sweep_results.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['type', 'label', 'psnr', 'lpips', 'time'])
        writer.writeheader()
        writer.writerows(results)

if __name__ == '__main__':
    run_all()