import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

epoch_files = {
    180: 'sweep_results.csv',
    56: 'marco_tests/test_res/sweep_results_x4_epoch50.csv',
    # 110: '...',
}

epoch_colors = {
    56:  '#0072B2',   # blue
    110: '#D55E00',   # orange
    180: '#009E73',   # green
}

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

all_time = []
all_psnr = []

for epoch, path in epoch_files.items():
    df = pd.read_csv(path)
    color = epoch_colors.get(epoch)

    # =========================
    # Fixed
    # =========================
    fixed = df[
        df['type'].astype(str).str.lower() == 'fixed'
    ].copy()

    fixed['num_pred'] = (
        fixed['label']
        .astype(str)
        .str.extract(r'num_pred=(\d+)')[0]
        .astype(float)
    )

    fixed = fixed.sort_values('num_pred')

    # Graph 1: fixed PSNR vs num_pred
    axes[0].plot(
        fixed['num_pred'],
        fixed['psnr'],
        'o-',
        color=color,
        linewidth=2,
        markersize=6,
        label=f'epoch {epoch}'
    )

    # Graph 2: fixed PSNR vs time ONLY
    axes[1].plot(
        fixed['time'],
        fixed['psnr'],
        'o-',
        color=color,
        linewidth=2,
        markersize=6,
        label=f'epoch {epoch}'
    )

    all_time.extend(fixed['time'].tolist())
    all_psnr.extend(fixed['psnr'].tolist())

    # =========================
    # Adaptive
    # =========================
    adaptive = df[
        df['type'].astype(str).str.lower().str.contains('adaptive')
    ].copy()

    # Robustly identify variance / sobel
    text = (
        adaptive['type'].astype(str) + ' ' +
        adaptive['label'].astype(str)
    ).str.lower()

    adaptive['metric'] = np.where(
        text.str.contains('sobel'),
        'sobel',
        'variance'
    )

    all_time.extend(adaptive['time'].tolist())
    all_psnr.extend(adaptive['psnr'].tolist())

    # =========================
    # Graph 3: adaptive only
    # =========================

    # ---- Variance ----
    variance = adaptive[
        adaptive['metric'] == 'variance'
    ].copy()

    if not variance.empty:
        axes[2].plot(
            variance['time'],
            variance['psnr'],
            's-',
            color=color,
            linewidth=2,
            markersize=7,
            label=f'epoch {epoch} variance'
        )

    # ---- Sobel ----
    sobel = adaptive[
        adaptive['metric'] == 'sobel'
    ].copy()

    if not sobel.empty:
        axes[2].plot(
            sobel['time'],
            sobel['psnr'],
            '^--',
            color=color,
            linewidth=2,
            markersize=7,
            label=f'epoch {epoch} sobel'
        )

        # Budget labels — Sobel only
        for _, row in sobel.iterrows():
            budgets = row['label'].split('[')[-1].split(']')[0]

            axes[2].annotate(
                budgets,
                (row['time'], row['psnr']),
                xytext=(0, 8),
                textcoords='offset points',
                ha='center',
                fontsize=7,
                alpha=0.85
            )


# ============================================================
# Graph 1
# ============================================================
axes[0].set_xscale('log', base=2)
axes[0].set_xlabel('num_pred')
axes[0].set_ylabel('PSNR (dB)')
axes[0].set_title('Fixed inference: PSNR vs num_pred')
axes[0].grid(alpha=0.2)
axes[0].legend(fontsize=8)


# ============================================================
# Graph 2
# ============================================================
axes[1].set_xlabel('Inference time (s)')
axes[1].set_ylabel('PSNR (dB)')
axes[1].set_title('Fixed inference: PSNR vs time')
axes[1].grid(alpha=0.2)
axes[1].legend(fontsize=8)


# ============================================================
# Graph 3
# ============================================================
axes[2].set_xlabel('Inference time (s)')
axes[2].set_ylabel('PSNR (dB)')
axes[2].set_title('Adaptive inference: PSNR vs time')
axes[2].grid(alpha=0.2)
axes[2].legend(fontsize=8)


# ============================================================
# Same scale for graphs 2 + 3
# ============================================================
xmin, xmax = min(all_time), max(all_time)
ymin, ymax = min(all_psnr), max(all_psnr)

xpad = (xmax - xmin) * 0.06
ypad = (ymax - ymin) * 0.08

axes[1].set_xlim(xmin - xpad, xmax + xpad)
axes[2].set_xlim(xmin - xpad, xmax + xpad)

axes[1].set_ylim(ymin - ypad, ymax + ypad)
axes[2].set_ylim(ymin - ypad, ymax + ypad)

axes[2].set_xticks(axes[1].get_xticks())
axes[2].set_yticks(axes[1].get_yticks())


plt.tight_layout()
plt.savefig('epoch_trend.png', dpi=200, bbox_inches='tight')
plt.show()

print("Saved: epoch_trend.png")