"""Make a recruiter-facing chart from redistributed aggregate research data."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "docs" / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)


def annualized_by_group(filename: str, specification: str) -> pd.Series:
    frame = pd.read_csv(RESULTS / filename)
    if "specification" in frame:
        frame = frame.loc[frame.specification.eq(specification)]
    wide = frame.pivot(index="signal_date", columns="group", values="net_return")
    return wide.mean().sort_index() * 12 * 100


def draw() -> Path:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": "#8a929e",
        "axes.labelcolor": "#273345",
        "xtick.color": "#4f5a6b",
        "ytick.color": "#4f5a6b",
        "text.color": "#1d2a3a",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), sharey=True)
    pairs = [
        (
            "all_stateful_decile_net_returns.csv",
            [("R0_residual_level", "Residual PB-ROE", "#2563a6"),
             ("R1_level_change", "Level + 1M change", "#e28d2c")],
            "A. Level + change",
            "171 common months",
        ),
        (
            "common_stateful_decile_net_returns.csv",
            [("pbroe_common", "Residual PB-ROE", "#2563a6"),
             ("combined_50_50", "PB-ROE + EPS revision", "#247d70")],
            "B. Analyst revision blend",
            "172 months, separate matched sample",
        ),
    ]
    for axis, (filename, specs, title, subtitle) in zip(axes, pairs):
        for name, label, color in specs:
            values = annualized_by_group(filename, name)
            axis.plot(values.index, values.values, marker="o", linewidth=2.2,
                      markersize=5.5, label=label, color=color)
        axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=22)
        axis.text(0, 1.01, subtitle, transform=axis.transAxes, fontsize=9,
                  color="#697586", va="bottom")
        axis.set_xticks(range(1, 11))
        axis.set_xlabel("Factor decile (low to high)")
        axis.grid(axis="y", color="#e8ebef", linewidth=0.7)
        axis.legend(frameon=False, loc="upper left", fontsize=9)
    axes[0].set_ylabel("After-cost annualized arithmetic return (%)")
    fig.suptitle("PB-ROE decile profiles", x=0.06, y=1.02, ha="left",
                 fontsize=17, fontweight="bold")
    fig.text(0.06, -0.02,
             "Monthly net group return × 12. Panels use different matched stock-month samples; "
             "compare each enhancement only with its own baseline.",
             fontsize=8.8, color="#697586")
    fig.tight_layout(w_pad=3.0)
    path = FIGURES / "decile_profiles.png"
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


if __name__ == "__main__":
    print(draw())
