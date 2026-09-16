import pandas as pd
import matplotlib.pyplot as plt
import ast

CSV_PATH = "sweep_results.csv"


def parse_budget(label):
    if label.startswith("num_pred="):
        return (int(label.split("=")[1]),)

    if label.startswith("budgets="):
        return tuple(ast.literal_eval(label.split("=", 1)[1]))

    raise ValueError(f"Unknown label format: {label}")


# =========================
# Load data
# =========================
df = pd.read_csv(CSV_PATH)
df["budget_tuple"] = df["label"].apply(parse_budget)

fixed = df[df["type"] == "fixed"].copy()
sobel = df[df["type"] == "adaptive-sobel"].copy()
variance = df[df["type"] == "adaptive-variance"].copy()
dct = df[df["type"] == "adaptive-dct"].copy()


# =========================
# Plot
# =========================
fig, ax = plt.subplots(figsize=(11, 7))

# Fixed
ax.scatter(
    fixed["time"],
    fixed["psnr"],
    s=70,
    label="Fixed"
)

# Adaptive methods
adaptive_data = {
    "Sobel": sobel,
    "Variance": variance,
    "DCT": dct,
}

for name, data in adaptive_data.items():

    ax.scatter(
        data["time"],
        data["psnr"],
        s=70,
        label=f"Adaptive ({name})"
    )

    # Budget labels
    for _, row in data.iterrows():
        budget_str = str(row["budget_tuple"]).replace("(", "").replace(")", "")

        ax.annotate(
            budget_str,
            (row["time"], row["psnr"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8
        )


# Fixed labels
for _, row in fixed.iterrows():
    budget = row["budget_tuple"][0]

    ax.annotate(
        str(budget),
        (row["time"], row["psnr"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8
    )


# =========================
# Compare methods
# =========================

merged = (
    sobel[["budget_tuple", "psnr", "time"]]
    .rename(columns={"psnr": "psnr_sobel", "time": "time_sobel"})
    .merge(
        variance[["budget_tuple", "psnr", "time"]]
        .rename(columns={
            "psnr": "psnr_variance",
            "time": "time_variance"
        }),
        on="budget_tuple",
        how="inner"
    )
    .merge(
        dct[["budget_tuple", "psnr", "time"]]
        .rename(columns={
            "psnr": "psnr_dct",
            "time": "time_dct"
        }),
        on="budget_tuple",
        how="inner"
    )
)

mean_psnr_var = (
    merged["psnr_variance"] - merged["psnr_sobel"]
).mean()

mean_time_var = (
    merged["time_variance"] - merged["time_sobel"]
).mean()

mean_psnr_dct = (
    merged["psnr_dct"] - merged["psnr_sobel"]
).mean()

mean_time_dct = (
    merged["time_dct"] - merged["time_sobel"]
).mean()


# =========================
# Summary box — upper left
# =========================

summary = (
    f"Sobel vs Variance\n"
    f"ΔPSNR: {mean_psnr_var:+.4f} dB\n"
    f"ΔTime: {mean_time_var:+.2f} s\n\n"
    f"Sobel vs DCT\n"
    f"ΔPSNR: {mean_psnr_dct:+.4f} dB\n"
    f"ΔTime: {mean_time_dct:+.2f} s"
)

ax.text(
    0.02,
    0.98,
    summary,
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(
        boxstyle="round",
        alpha=0.85
    )
)


# =========================
# Formatting
# =========================

ax.set_xlabel("Inference Time (s)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("PSNR vs Inference Time")

ax.legend(loc="lower right")

ax.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    "sweep_psnr_vs_time.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()