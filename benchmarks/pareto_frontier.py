import pandas as pd

FILE = "sweep_preliminary.csv"


def pareto_frontier(df):
    """
    Pareto frontier for:
        time: minimize
        psnr: maximize
    """
    rows = []

    for i, row in df.iterrows():
        dominated = (
            (df["time"] <= row["time"]) &
            (df["psnr"] >= row["psnr"]) &
            (
                (df["time"] < row["time"]) |
                (df["psnr"] > row["psnr"])
            )
        ).any()

        if not dominated:
            rows.append(i)

    return df.loc[rows].sort_values("time")


df = pd.read_csv(FILE)

# ------------------------------------------------------------
# Overall Pareto frontier
# ------------------------------------------------------------

frontier = pareto_frontier(df)

print("\n=== OVERALL PARETO FRONTIER ===")
print(
    frontier[
        ["type", "label", "psnr", "lpips", "time"]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# Adaptive configurations on the frontier
# ------------------------------------------------------------

adaptive_frontier = frontier[
    frontier["type"].str.startswith("adaptive")
]

print("\n=== ADAPTIVE CONFIGURATIONS ON FRONTIER ===")

for method, group in adaptive_frontier.groupby("type"):
    print(f"\n{method}")
    print(
        group[
            ["label", "psnr", "lpips", "time"]
        ].to_string(index=False)
    )


# ------------------------------------------------------------
# All adaptive configurations that are NOT dominated by a
# FIXED model.
#
# These are potentially interesting even if another adaptive
# method currently beats them.
# ------------------------------------------------------------

fixed = df[df["type"] == "fixed"]
adaptive = df[df["type"].str.startswith("adaptive")]

interesting = []

for i, row in adaptive.iterrows():

    dominated_by_fixed = (
        (fixed["time"] <= row["time"]) &
        (fixed["psnr"] >= row["psnr"]) &
        (
            (fixed["time"] < row["time"]) |
            (fixed["psnr"] > row["psnr"])
        )
    ).any()

    if not dominated_by_fixed:
        interesting.append(i)

interesting = adaptive.loc[interesting].sort_values("time")

print("\n=== ADAPTIVE POINTS NOT DOMINATED BY FIXED ===")
print(
    interesting[
        ["type", "label", "psnr", "lpips", "time"]
    ].to_string(index=False)
)


# ------------------------------------------------------------
# For each adaptive method separately:
# find its own Pareto frontier.
# ------------------------------------------------------------

print("\n=== FRONTIER PER METHOD ===")

for method in sorted(adaptive["type"].unique()):
    group = adaptive[adaptive["type"] == method]
    pf = pareto_frontier(group)

    print(f"\n{method}")
    print(
        pf[
            ["label", "psnr", "lpips", "time"]
        ].to_string(index=False)
    )