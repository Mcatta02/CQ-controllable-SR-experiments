import argparse
import os
import numpy as np
from tqdm import tqdm
import yaml
import math
from functools import partial

import torch
import utils
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, BatchSampler
from difficulty import compute_tile_difficulty, difficulty_to_budget, budget_map_to_coord_order
import torchvision

from lpips import LPIPS

import models
import losses
import datasets
import utils

MAX_ENCODER_PIXELS = 2_766_240

def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', type=str, default='', metavar='FILE', help='')
    parser.add_argument('--save_dir', type=str, default='')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--gpu', type=str, default='0', help='gpu ids to use')
    parser.add_argument('--model', type=str, default='')
    parser.add_argument('--num_pred', type=int, default=None)
    parser.add_argument('--save_img', action='store_true', help='save images to the save_dir')
    parser.add_argument('--save_fourier', action='store_true', help='save fourier features to the save_dir')
    parser.add_argument('--adaptive', action='store_true', help='use tile-adaptive num_pred instead of fixed')
    parser.add_argument('--tile_size', type=int, default=8)
    parser.add_argument('--budgets', type=int, nargs='+', default=[4, 8, 16, 32, 64])
    
    return parser.parse_args()
def test():
    timer = utils.Timer()
    print('Loading Datasets...')
    if config.get('data_norm'):
        utils.set_normalizer(**config['data_norm'])
    dataset_spec = config['test_dataset']
    dataset = datasets.make(dataset_spec['dataset'])
    dataset = datasets.make(dataset_spec['wrapper'], args={'dataset': dataset})
    data_loader = DataLoader(dataset, batch_size=dataset_spec['batch_size'],\
        num_workers=args.num_workers, pin_memory=True)
    print(f'Done. It took {utils.time_text(timer.t())}')

    timer.s()
    print('Preparing Model...')
    model_spec = torch.load(args.model)['model']
    model = models.make(model_spec, load_sd=True).cuda()
    if args.num_pred is not None:
        model.predictor.num_pred = args.num_pred
        model.num_pred = args.num_pred
        model.num_preds = args.num_pred
    print(f'Done. It took {utils.time_text(timer.t())}')

    num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if num_gpus > 1:
        model = nn.DataParallel(model, device_ids=list(range(num_gpus)))

    psnr, lpips, inference_time = do_test(model, data_loader, save_dir=save_path, batch_size=config.get('eval_bsize'))
    print(f'PSNR: {psnr}, LPIPS: {lpips}, Inference Time: {inference_time}')

    # output results to the txt file
    with open(os.path.join(save_path, 'results.txt'), 'w') as f:
        f.write(f'PSNR: {psnr}, LPIPS: {lpips}, Inference Time: {inference_time}\n')

def do_test(model, data_loader, save_dir=None, batch_size=None, validation=False, lpips_net=None):
    model.eval()

    if not validation:
        if config['eval_type'] is None:
            psnr_fn = utils.calc_psnr
        elif config['eval_type'].startswith('div2k'):
            scale = int(config['eval_type'].split('-')[1])
            psnr_fn = partial(utils.calc_psnr, dataset='div2k', scale=scale)
        elif config['eval_type'].startswith('benchmark'):
            scale = int(config['eval_type'].split('-')[1])
            psnr_fn = partial(utils.calc_psnr, dataset='benchmark', scale=scale)
        else:
            raise NotImplementedError
    else:
        psnr_fn = utils.calc_psnr

    psnr = utils.Avarager()
    lpips = utils.Avarager()
    inference_time = utils.Avarager()
    timer = utils.Timer()
    if not validation and lpips_net is None:
        lpips_net = LPIPS(net='alex').cuda()

    skipped = 0
    processed = 0

    for i, (inputs, targets) in enumerate(tqdm(data_loader)):

        # Check input resolution before expensive encoder inference
        inp = inputs['inp']
        _, _, h, w = inp.shape

        if h * w > MAX_ENCODER_PIXELS:
            skipped += 1
            tqdm.write(
                f"[SKIP img {i}] Input too large for full-image SwinIR inference: "
                f"{h}x{w} ({h*w:,} pixels > {MAX_ENCODER_PIXELS:,})"
            )
            torch.cuda.empty_cache()
            continue

        torch.cuda.empty_cache()

        inputs = {k: v.cuda() for k, v in inputs.items()}
        targets = {k: v.cuda() for k, v in targets.items()}

        timer.s()

        with torch.no_grad():
            if batch_size is None:
                preds = model(inputs)
            else:
                preds = batched_predict(
                    model,
                    inputs,
                    batch_size,
                    adaptive=args.adaptive,
                    tile_size=args.tile_size,
                    budgets=args.budgets,
                    metric=getattr(args, 'metric', 'variance')
                )

        inference_time.add(timer.t())

        if validation:
            preds['recon'] = utils.denormalize(preds['recon']).clamp_(0, 1)
            targets['gt_rgb'] = utils.denormalize(targets['gt_rgb'])
            psnr.add(psnr_fn(preds['recon'], targets['gt_rgb']).item(), inputs['inp'].shape[0])

        else:
            preds['recon'] = preds['recon'].cuda()
            ih, iw = inputs['inp'].shape[-2:]
            s = math.sqrt(inputs['coord'].shape[1] / (ih * iw))
            shape = [inputs['inp'].shape[0], round(ih * s), round(iw * s), -1]
            preds['recon'] = preds['recon'].view(*shape).permute(0, 3, 1, 2).contiguous()
            preds['recon'] = preds['recon'][..., :targets['gt_img'].shape[-2], :targets['gt_img'].shape[-1]]

            preds['recon'] = utils.denormalize(preds['recon']).clamp_(0, 1)
            for j in range(preds['recon'].shape[0]):
                preds['recon'][j] = utils.discretize(preds['recon'][j])
            preds['recon'] = utils.normalize(preds['recon'])

            with torch.no_grad():
                lpips.add(lpips_net(preds['recon'], targets['gt_img']).mean().item(), inputs['inp'].shape[0])

            preds['recon'] = utils.denormalize(preds['recon']).clamp_(0, 1)
            targets['gt_img'] = utils.denormalize(targets['gt_img'])

            res = psnr_fn(preds['recon'], targets['gt_img'])
            psnr.add(res.item(), inputs['inp'].shape[0])

            if (save_dir is not None) and args.save_img:
                for j in range(preds['recon'].shape[0]):
                    save_imgs(preds['recon'][j], os.path.join(save_dir, f'pred_{str(i).zfill(4)}_{res.item():.2f}.png'))
                    if args.save_fourier:
                        save = {k: v for k, v in preds.items()}
                        torch.save(save, os.path.join(save_dir, f'preds_{i}.pth'))
    
        processed += 1

    print(
        f"Processed: {processed}/{processed + skipped} | "
        f"Skipped: {skipped}"
    )

    return psnr.item(), lpips.item(), inference_time.item()

def batched_predict(model, inputs, batch_size, adaptive=False, tile_size=8, budgets=None, metric='variance'):
    with torch.no_grad():
        inp, coord, cell = inputs['inp'], inputs['coord'], inputs['cell']

        model.gen_feat(inp)

        n = coord.shape[1]

        if not adaptive:
            ql = 0
            preds = {}
            while ql < n:
                qr = min(ql + batch_size, n)
                pred = model.query(coord[:, ql:qr], cell[:, ql:qr])
                for k, v in pred.items():
                    preds.setdefault(k, [])
                    preds[k].append(v.cpu() if torch.is_tensor(v) else [x.cpu() for x in v])
                ql = qr
            for k, v in preds.items():
                if torch.is_tensor(v[0]):
                    preds[k] = torch.cat(v, dim=1)
                else:
                    preds[k] = [torch.cat([x[i] for x in v], dim=1) for i in range(len(v[0]))]
            return preds

        # adaptive: bucket ONCE over the whole image, chunk within each bucket for memory only
        ih, iw = inp.shape[-2:]
        s = math.sqrt(n / (ih * iw))
        out_h, out_w = round(ih * s), round(iw * s)
        diff = compute_tile_difficulty(inp, tile_size=tile_size, metric=metric)
        bmap = difficulty_to_budget(diff, budgets=budgets)
        budget = budget_map_to_coord_order(bmap, out_h, out_w).to(inp.device)  # [1, n]

        assert coord.shape[0] == 1, "adaptive batched_predict currently assumes batch_size 1"
        budget_flat = budget[0]
        recon = torch.zeros(1, n, model.decoder.out_dim)

        for b in budget_flat.unique().tolist():
            idx = (budget_flat == b).nonzero(as_tuple=True)[0]
            ql = 0
            while ql < idx.shape[0]:
                qr = min(ql + batch_size, idx.shape[0])
                sub_idx = idx[ql:qr]
                out = model.adaptive_query(coord[:, sub_idx], cell[:, sub_idx], budget[:, sub_idx])
                recon[:, sub_idx] = out['recon'].cpu()
                ql = qr

        return {'recon': recon, 'budget': budget.cpu()}

def save_imgs(img, path):
    img = torchvision.transforms.ToPILImage()(img.cpu())
    img.save(path)

def main():
    global args, config, save_path
    args = parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # set gpu environment
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        # torch.backends.cudnn.deterministic = True
    else:
        raise Exception('GPU not found.')

    # set save path
    save_name = args.save_dir
    assert len(save_name) > 0, 'save_dir must be specified.'
    # save_path = os.path.join('./save', save_name)
    save_path = save_name

    utils.ensure_path(save_path, remove=True)

    test()

if __name__ == '__main__':
    main()
