import itertools

import torch
from torch import nn
from torch.nn import functional as F

from models import register, make
from utils import make_coord, fourier_decoding
import utils

@register('liif-base')
class LIIFBase(nn.Module):
    def __init__(self, encoder_spec, predictor_spec, decoder_spec, feat_unfold=False, local_ensemble=False, residual=True):
        super().__init__()

        self.feat_unfold = feat_unfold
        self.local_ensemble = local_ensemble
        self.residual = residual

        self.encoder = make(encoder_spec)
        self.decoder = make(decoder_spec)
        self.predictor = make(predictor_spec, args={'in_dim': self.encoder.out_dim})
        num_params = utils.compute_num_params(self.predictor, text=False)
        # print(f'Estimated memory consumption of predictor: {num_params*32/1024}MB')
        self.num_pred = self.predictor.num_pred if hasattr(self.predictor, 'num_pred') else 1
        self.num_preds = self.num_pred

    def forward(self, inp, coord, cell=None, **kwargs):
        self.gen_feat(inp)
        return self.query(coord, cell)

    def gen_feat(self, x):
        self.inp = x
        self.feat = self.encoder(x)
        self.feat_coord = make_coord(x.shape[-2:], flatten=False).to(x.device) \
            .permute(2, 0, 1).unsqueeze(0).expand(x.shape[0], 2, *x.shape[-2:])

    def query(self, coord, cell=None):
        feat = self.feat
        feat_coord = self.feat_coord

        # fold the 8 neighboring features into the center feature
        if self.feat_unfold:
            feat = F.unfold(feat, 3, padding=1).view(\
                feat.shape[0], feat.shape[1] * 9, feat.shape[2], feat.shape[3])

        # setting up the local ensemble
        if self.local_ensemble:
            vx_lst = [-1, 1]
            vy_lst = [-1, 1]
            eps_shift = 1e-6
        else:
            vx_lst, vy_lst, eps_shift = [0], [0], 0

        # field radius (global: [-1, 1])
        rx = 2 / feat.shape[-2] / 2
        ry = 2 / feat.shape[-1] / 2

        preds, areas, miscs = [], [], {}
        for vx, vy in itertools.product(vx_lst, vy_lst):
            coord_ = coord.clone()
            coord_[:, :, 0] += vx * rx + eps_shift
            coord_[:, :, 1] += vy * ry + eps_shift
            coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

            q_feat  = F.grid_sample(feat, coord_.flip(-1).unsqueeze(1), \
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
            q_coord = F.grid_sample(feat_coord, coord_.flip(-1).unsqueeze(1), \
                mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

            rel_coord = coord - q_coord
            rel_coord[:, :, 0] *= feat.shape[-2]
            rel_coord[:, :, 1] *= feat.shape[-1]

            rel_cell = cell.clone()
            rel_cell[:, :, 0] *= feat.shape[-2]
            rel_cell[:, :, 1] *= feat.shape[-1]

            bs, q = q_feat.shape[:2]
            q_feat, rel_cell, rel_coord = q_feat.reshape(bs * q, -1), rel_cell.reshape(bs * q, -1), rel_coord.reshape(bs * q, -1)

            latent = self.predictor(q_feat, rel_cell, rel_coord)
            out = self.decoder(latent, scale=self.num_pred)
            # out = self.decoder(latent)

            q_feat, rel_cell, rel_coord = q_feat.view(bs, q, -1), rel_cell.view(bs, q, -1), rel_coord.view(bs, q, -1)
            out = {k: v.view(bs, q, *v.shape[1:]) for k, v in out.items()}

            out['rel_coord'] = rel_coord

            if self.num_pred != self.num_preds:
                out = self.partial_reconstruction(out, 'specify', self.num_preds, length=self.num_pred)

            preds.append(out['pred'])
            area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
            areas.append(area + 1e-9)

            for k, v in out.items():
                if k not in miscs:
                    miscs[k] = []
                miscs[k].append(v)

        ret = self.reconstruct_pixels(preds, areas, coord)
        miscs['inp'] = self.inp
        miscs['area'] = areas
        miscs['coord'] = coord

        self._debug = {'q_feat': q_feat.clone(), 'rel_coord': rel_coord.clone(),
                        'rel_cell': rel_cell.clone(), 'latent': latent, 'pred_before_residual': preds[0].clone()}


        return {'recon': ret, **miscs}

    #forces exactly T recurrent steps and bypasses partial_reconstruction entirely
    def query_t(self, coord, cell, T):

        feat, feat_coord = self.feat, self.feat_coord

        q_feat = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]; rel_coord[:, :, 1] *= feat.shape[-1]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]; rel_cell[:, :, 1] *= feat.shape[-1]

        bs, q = q_feat.shape[:2]
        q_feat, rel_cell, rel_coord = (x.reshape(bs * q, -1) for x in (q_feat, rel_cell, rel_coord))

        self.predictor.num_pred = T
        latent = self.predictor(q_feat, rel_cell, rel_coord)
        pred = self.decoder(latent, scale=T)['pred']

        ret = pred.view(bs, q, -1)

        if self.residual:
            ret = ret + F.grid_sample(self.inp.to(coord.device), coord.flip(-1).unsqueeze(1),
                mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

            self._debug_t = {'q_feat': q_feat.clone(), 'rel_coord': rel_coord.clone(),
                            'rel_cell': rel_cell.clone(), 'latent': latent, 'pred_before_residual': pred.view(bs, q, -1).clone()}

        return ret

    def adaptive_query(self, coord, cell, budget):
        """budget: LongTensor [B, Q], same flatten order as coord. local_ensemble off for now."""
        feat, feat_coord = self.feat, self.feat_coord

        q_feat = F.grid_sample(feat, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        q_coord = F.grid_sample(feat_coord, coord.flip(-1).unsqueeze(1), mode='nearest', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        rel_coord = coord - q_coord
        rel_coord[:, :, 0] *= feat.shape[-2]; rel_coord[:, :, 1] *= feat.shape[-1]
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]; rel_cell[:, :, 1] *= feat.shape[-1]

        bs, q = q_feat.shape[:2]
        q_feat, rel_cell, rel_coord = (x.reshape(bs * q, -1) for x in (q_feat, rel_cell, rel_coord))
        budget_flat = budget.reshape(bs * q)

        pred = torch.zeros(bs * q, self.decoder.out_dim, device=feat.device, dtype=feat.dtype)
        for b in budget_flat.unique().tolist():
            idx = (budget_flat == b).nonzero(as_tuple=True)[0]
            self.predictor.num_pred = b
            latent = self.predictor(q_feat[idx], rel_cell[idx], rel_coord[idx])
            pred[idx] = self.decoder(latent, scale=b)['pred']

        ret = pred.view(bs, q, -1)
        if self.residual:
            ret = ret + F.grid_sample(self.inp.to(coord.device), coord.flip(-1).unsqueeze(1),
                mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)
        return {'recon': ret, 'budget': budget}

    def reconstruct_pixels(self, preds, areas, coord):
        total_area = torch.stack(areas).sum(dim=0)
        if self.local_ensemble:
            t = areas[0]; areas[0] = areas[3]; areas[3] = t
            t = areas[1]; areas[1] = areas[2]; areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret += pred * (area / total_area).unsqueeze(-1)

        if self.residual:
            residual = F.grid_sample(self.inp.to(coord.device), coord.flip(-1).unsqueeze(1), \
                mode='bilinear', padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

            # when num_pred == 0, return the input
            if ret.shape[-1] == 0:
                ret = residual
            else:
                ret += residual

        return ret

    def partial_reconstruction(self, pred, mode, num_pred, length=None):
        bs, q = pred['coef'].shape[:2]
        block_size = getattr(self.predictor, 'block_size', 1)

        coef, freq = pred['coef'].view(bs * q, -1), pred['freq'].view(bs * q, -1)
        phase, rel_coord = pred['phase'].view(bs * q, -1), pred['rel_coord'].view(bs * q, -1)

        fourier = self.predictor.reshape(coef, freq)

        if mode == 'random':
            length = torch.randint(num_pred, (bs * q, 1), device=coef.device) + 1
        elif mode == 'specify':
            length = length
        mask = torch.arange(num_pred, device=coef.device).expand(bs * q, num_pred) < length
        mask = mask.unsqueeze(-2).expand(bs * q, 2*6*block_size, num_pred)

        fourier = fourier * mask
        coef, freq = self.predictor.unreshape(fourier)

        decoded = fourier_decoding(coef, freq, phase, rel_coord)
        pred = self.decoder({'decoded': decoded}, scale=length)['pred'].view(bs, q, -1)
        out = {
            'decoded': decoded.view(bs, q, -1),
            'coef': coef.view(bs, q, -1),
            'freq': freq.view(bs, q, -1),
            'phase': phase.view(bs, q, -1),
            'pred': pred.view(bs, q, -1),
        }

        return out

    def partial_reconstruction_debug(self, pred, mode, num_pred, length=None):
        bs, q = pred['coef'].shape[:2]
        block_size = getattr(self.predictor, 'block_size', 1)

        coef, freq = pred['coef'].view(bs * q, -1), pred['freq'].view(bs * q, -1)
        phase, rel_coord = pred['phase'].view(bs * q, -1), pred['rel_coord'].view(bs * q, -1)

        fourier = self.predictor.reshape(coef, freq)

        if mode == 'random':
            length = torch.randint(num_pred, (bs * q, 1), device=coef.device) + 1
        elif mode == 'specify':
            length = length
        mask = torch.arange(num_pred, device=coef.device).expand(bs * q, num_pred) < length
        mask = mask.unsqueeze(-2).expand(bs * q, 2*6*block_size, num_pred)

        fourier = fourier * mask
        coef, freq = self.predictor.unreshape(fourier)

        decoded = fourier_decoding(coef, freq, phase, rel_coord)
        pred = self.decoder({'decoded': decoded}, scale=num_pred)['pred'].view(bs, q, -1)
        out = {
            'decoded': decoded,
            'coef': coef,
            'freq': freq,
            'phase': phase,
            'rel_coord': rel_coord,
            'pred': pred,
        }

        return out