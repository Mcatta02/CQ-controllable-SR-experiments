import torch
import torch.nn.functional as F

_SOBEL_X = None
_SOBEL_Y = None

def difficulty_to_budget(diff_map, budgets=(4, 8, 16, 32, 64)):
    """Per-image quantile buckets -> equal-mass groups regardless of image content."""
    flat = diff_map.flatten(1)
    qs = torch.linspace(0, 1, len(budgets) + 1, device=flat.device)[1:-1]
    thresholds = torch.quantile(flat, qs, dim=1)          # [len(budgets)-1, B]
    idx = torch.zeros_like(diff_map, dtype=torch.long)
    for t in thresholds:
        idx += (diff_map > t.view(-1, 1, 1)).long()
    budgets_t = torch.tensor(budgets, device=diff_map.device)
    return budgets_t[idx.clamp(max=len(budgets) - 1)]      # [B, nh, nw]

def budget_map_to_coord_order(budget_map, out_h, out_w):
    """Nearest-upsample to output resolution and flatten to match utils.make_coord's
    row-major (meshgrid) ordering — same order test.py uses when reshaping preds."""
    up = F.interpolate(budget_map.float().unsqueeze(1), size=(out_h, out_w), mode='nearest')
    return up.long().view(budget_map.shape[0], -1)

def _get_sobel_kernels(dtype, device):
    global _SOBEL_X, _SOBEL_Y
    if _SOBEL_X is None or _SOBEL_X.device != device or _SOBEL_X.dtype != dtype:
        _SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                                 dtype=dtype, device=device).view(1, 1, 3, 3)
        _SOBEL_Y = _SOBEL_X.transpose(-1, -2).contiguous()
    return _SOBEL_X, _SOBEL_Y


def compute_tile_difficulty_sobel(lr_img, tile_size=8):
    gray = lr_img.mean(dim=1, keepdim=True)
    sobel_x, sobel_y = _get_sobel_kernels(gray.dtype, gray.device)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    grad_mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-8)

    patches = F.unfold(grad_mag, kernel_size=tile_size, stride=tile_size)
    energy = patches.mean(dim=1)

    nh, nw = lr_img.shape[-2] // tile_size, lr_img.shape[-1] // tile_size
    return energy.view(lr_img.shape[0], nh, nw)


def compute_tile_difficulty(lr_img, tile_size=8, metric='variance'):
    """Dispatch wrapper — same call site, choose the metric by name."""
    if metric == 'variance':
        gray = lr_img.mean(dim=1, keepdim=True)
        patches = F.unfold(gray, kernel_size=tile_size, stride=tile_size)
        var = patches.var(dim=1, unbiased=False)
        nh, nw = lr_img.shape[-2] // tile_size, lr_img.shape[-1] // tile_size
        return var.view(lr_img.shape[0], nh, nw)
    elif metric == 'sobel':
        return compute_tile_difficulty_sobel(lr_img, tile_size=tile_size)
    elif metric == 'dct':
        return compute_tile_difficulty_dct(lr_img, tile_size=tile_size)
    else:
        raise ValueError(f"unknown metric: {metric}")


def compute_tile_difficulty_dct(lr_img, tile_size=8, freq_cutoff_ratio=0.5):
    """Block-DCT high-frequency energy per tile — directly measures how much
    high-frequency content a tile has, which is what num_pred is meant to capture."""
    gray = lr_img.mean(dim=1, keepdim=True)  # [B,1,H,W]
    B, _, H, W = gray.shape
    nh, nw = H // tile_size, W // tile_size

    patches = F.unfold(gray, kernel_size=tile_size, stride=tile_size)  # [B, tile_size^2, L]
    patches = patches.transpose(1, 2).reshape(-1, tile_size, tile_size)  # [B*L, ts, ts]

    # 2D DCT via two 1D DCTs (matrix form) — build once, cache like the Sobel kernel
    dct_mat = _get_dct_matrix(tile_size, patches.dtype, patches.device)
    dct = dct_mat @ patches @ dct_mat.transpose(-1, -2)  # [B*L, ts, ts]

    # zero out the low-frequency (top-left) block, keep the rest as "high-frequency energy"
    cutoff = max(1, int(tile_size * freq_cutoff_ratio))
    mask = torch.ones_like(dct)
    mask[:, :cutoff, :cutoff] = 0
    energy = (dct * mask).pow(2).sum(dim=(-1, -2))  # [B*L]

    return energy.view(B, nh, nw)


_DCT_MAT = None

def _get_dct_matrix(n, dtype, device):
    global _DCT_MAT
    if _DCT_MAT is None or _DCT_MAT.shape[0] != n or _DCT_MAT.device != device or _DCT_MAT.dtype != dtype:
        k = torch.arange(n, dtype=dtype, device=device).view(-1, 1)
        i = torch.arange(n, dtype=dtype, device=device).view(1, -1)
        _DCT_MAT = torch.cos(torch.pi / n * (i + 0.5) * k)
        _DCT_MAT[0] *= 1 / torch.sqrt(torch.tensor(2.0, dtype=dtype, device=device))
        _DCT_MAT *= torch.sqrt(torch.tensor(2.0 / n, dtype=dtype, device=device))
    return _DCT_MAT