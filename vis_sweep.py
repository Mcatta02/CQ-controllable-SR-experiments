import pandas as pd
import matplotlib.pyplot as plt
import ast

# ---------------------------------------------------------
# Load results
# ---------------------------------------------------------

CSV_PATH = "sweep_results.csv"

df = pd.read_csv(CSV_PATH)


# ---------------------------------------------------------
# Parse labels into comparable budget tuples
# ---------------------------------------------------------

def parse_budget(label):
    """
    Convert:
        num_pred=4       -> (4,)
        budgets=[4, 8]   -> (4, 8)
    """
    if label.startswith("num_pred="):
        return (int(label.replace("num_pred=", "")),)

    budgets = label.replace("budgets=", "")
    return tuple(ast.literal_eval(budgets))


df["budget_tuple"] = df["label"].apply(parse_budget)


# ---------------------------------------------------------
# Separate methods
# ---------------------------------------------------------

fixed = df[df["type"] == "fixed"].copy()

sobel = (
    df[df["type"] == "adaptive-sobel"]
    .copy()
    .set_index("budget_tuple")
)

variance = (
    df[df["type"] == "adaptive-variance"]
    .copy()
    .set_index("budget_tuple")
)


# ---------------------------------------------------------
# Match Sobel and Variance results
# ---------------------------------------------------------

common_budgets = sobel.index.intersection(variance.index)

comparison = pd.DataFrame({
    "sobel_psnr": sobel.loc[common_budgets, "psnr"],
    "variance_psnr": variance.loc[common_budgets, "psnr"],
    "sobel_time": sobel.loc[common_budgets, "time"],
    "variance_time": variance.loc[common_budgets, "time"],
})

comparison["delta_psnr"] = (
    comparison["sobel_psnr"] - comparison["variance_psnr"]
)

comparison["delta_time"] = (
    comparison["sobel_time"] - comparison["variance_time"]
)


# ---------------------------------------------------------
# Average difference
# ---------------------------------------------------------

mean_delta_psnr = comparison["delta_psnr"].mean()
mean_delta_time = comparison["delta_time"].mean()

print("Sobel vs Variance")
print(f"Mean ΔPSNR: {mean_delta_psnr:+.4f} dB")
print(f"Mean ΔTime: {mean_delta_time:+.4f} s")


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

fig, ax = plt.subplots(figsize=(12, 7))

# Fixed
ax.scatter(
    fixed["time"],
    fixed["psnr"],
    s=65,
    label="Fixed",
)

# Adaptive Sobel
ax.scatter(
    sobel["time"],
    sobel["psnr"],
    s=60,
    label="Adaptive - Sobel",
)

# Adaptive Variance
ax.scatter(
    variance["time"],
    variance["psnr"],
    s=60,
    label="Adaptive - Variance",
)


# ---------------------------------------------------------
# Labels
# ---------------------------------------------------------

# Fixed labels
for _, row in fixed.iterrows():

    label = row["label"].replace("num_pred=", "")

    ax.annotate(
        label,
        (row["time"], row["psnr"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
    )


# Only label Sobel adaptive points.
# Variance points use the same labels, so there is no need
# to print them twice.
for _, row in sobel.reset_index().iterrows():

    label = ",".join(map(str, row["budget_tuple"]))

    ax.annotate(
        label,
        (row["time"], row["psnr"]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=8,
    )


# ---------------------------------------------------------
# Summary box
# ---------------------------------------------------------

summary = (
    f"Sobel − Variance\n"
    f"Mean ΔPSNR: {mean_delta_psnr:+.4f} dB\n"
    f"Mean ΔTime: {mean_delta_time:+.4f} s"
)

ax.text(
    0.98,
    0.18,
    summary,
    transform=ax.transAxes,
    ha="right",
    va="bottom",
    fontsize=9,
    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        alpha=0.85,
        edgecolor="gray",
    ),
)


# ---------------------------------------------------------
# Formatting
# ---------------------------------------------------------

ax.set_xlabel("Inference time (s)")
ax.set_ylabel("PSNR (dB)")
ax.set_title("PSNR vs. Inference Time")

ax.set_ylim(31.3, 31.4)

ax.grid(True, alpha=0.25)

# Legend in bottom-right
ax.legend(
    loc="lower right",
)

plt.tight_layout()

plt.savefig(
    "psnr_vs_time.png",
    dpi=300,
    bbox_inches="tight",
)

plt.show()