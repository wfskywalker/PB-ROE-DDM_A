"""Recompute selected published metrics from redistributed aggregate series."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def close(actual: float, expected: float, label: str) -> None:
    if not np.isclose(actual, expected, atol=1e-10, rtol=1e-9):
        raise AssertionError(f"{label}: recomputed {actual}, published {expected}")


def ic_metrics(values: pd.Series) -> tuple[float, float]:
    series = values.astype(float)
    return float(series.mean()), float(np.sqrt(12) * series.mean() / series.std(ddof=1))


def decile_metrics(frame: pd.DataFrame, name: str) -> tuple[float, float, int]:
    selected = frame.loc[frame.specification.eq(name)] if "specification" in frame else frame
    wide = selected.pivot(
        index="signal_date", columns="group", values="net_return"
    ).sort_index()
    if wide.isna().any().any() or list(wide.columns) != list(range(1, 11)):
        raise AssertionError(f"{name}: incomplete ten-group monthly series")
    spread = float((wide[10] - wide[1]).mean() * 12)
    means = wide.mean()
    rho = float(pd.Series(range(1, 11), index=means.index).corr(means, method="spearman"))
    adjacent = int((means.diff().iloc[1:] > 0).sum())
    return spread, rho, adjacent


def spread_sharpe(frame: pd.DataFrame, name: str) -> float:
    selected = frame.loc[frame.specification.eq(name)] if "specification" in frame else frame
    wide = selected.pivot(index="signal_date", columns="group", values="net_return")
    spread = wide[10] - wide[1]
    return float(np.sqrt(12) * spread.mean() / spread.std(ddof=1))


def verify_standalone() -> None:
    ic = pd.read_csv(RESULTS / "monthly_rank_ic_comparison.csv")
    summary = pd.read_csv(RESULTS / "raw_vs_neutralized_summary.csv").set_index("specification")
    net = pd.read_csv(RESULTS / "neutralized_stateful_decile_net_returns.csv")
    mean, ir = ic_metrics(ic.neutralized_rank_ic)
    close(mean, summary.loc["neutralized", "mean_rank_ic"], "standalone mean IC")
    close(ir, summary.loc["neutralized", "rank_icir"], "standalone ICIR")
    spread, _, _ = decile_metrics(net, "neutralized")
    close(spread, summary.loc["neutralized", "annualized_arithmetic_g10_g1"],
          "standalone after-cost spread")
    print(f"Standalone: {len(ic)} months; mean Rank IC {mean:.5f}; ICIR {ir:.3f}; spread {spread:.2%}")


def verify_level_change() -> None:
    ic = pd.read_csv(RESULTS / "all_monthly_rank_ic.csv")
    summary = pd.read_csv(RESULTS / "all_summary.csv").set_index("specification")
    risk = pd.read_csv(RESULTS / "all_risk.csv")
    mono = pd.read_csv(RESULTS / "all_monotonicity.csv").set_index("specification")
    net = pd.read_csv(RESULTS / "all_stateful_decile_net_returns.csv")
    for name in ("R0_residual_level", "R1_level_change"):
        mean, ir = ic_metrics(ic[f"{name}_rank_ic"])
        spread, rho, adjacent = decile_metrics(net, name)
        close(mean, summary.loc[name, "mean_rank_ic"], f"{name} mean IC")
        close(ir, summary.loc[name, "rank_icir"], f"{name} ICIR")
        close(spread, summary.loc[name, "annualized_arithmetic_g10_g1"],
              f"{name} after-cost spread")
        close(rho, mono.loc[name, "full_history_group_spearman"], f"{name} group rho")
        published_sharpe = risk.loc[
            risk.specification.eq(name) & risk.portfolio.eq("G10-G1"),
            "sharpe_zero_rf",
        ].iloc[0]
        close(spread_sharpe(net, name), published_sharpe, f"{name} spread Sharpe")
        assert adjacent == int(mono.loc[name, "adjacent_pairs_in_order"])
        print(f"{name}: {len(ic)} months; IC {mean:.5f}; ICIR {ir:.3f}; "
              f"spread {spread:.2%}; adjacent {adjacent}/9")


def verify_revision_blend() -> None:
    ic = pd.read_csv(RESULTS / "common_monthly_ic_correlation.csv")
    summary = pd.read_csv(RESULTS / "common_specification_summary.csv").set_index("specification")
    risk = pd.read_csv(RESULTS / "common_specification_risk.csv")
    mono = pd.read_csv(RESULTS / "common_specification_monotonicity.csv").set_index("specification")
    net = pd.read_csv(RESULTS / "common_stateful_decile_net_returns.csv")
    for name, column in (
        ("pbroe_common", "pbroe_rank_ic"),
        ("combined_50_50", "combined_rank_ic"),
    ):
        mean, ir = ic_metrics(ic[column])
        spread, rho, adjacent = decile_metrics(net, name)
        close(mean, summary.loc[name, "mean_rank_ic"], f"{name} mean IC")
        close(ir, summary.loc[name, "rank_icir"], f"{name} ICIR")
        close(spread, summary.loc[name, "annualized_arithmetic_g10_g1"],
              f"{name} after-cost spread")
        close(rho, mono.loc[name, "full_history_group_spearman"], f"{name} group rho")
        published_sharpe = risk.loc[
            risk.specification.eq(name) & risk.portfolio.eq("G10-G1"),
            "sharpe_zero_rf",
        ].iloc[0]
        close(spread_sharpe(net, name), published_sharpe, f"{name} spread Sharpe")
        assert adjacent == int(mono.loc[name, "adjacent_pairs_in_order"])
        print(f"{name}: {len(ic)} months; IC {mean:.5f}; ICIR {ir:.3f}; "
              f"spread {spread:.2%}; adjacent {adjacent}/9")


if __name__ == "__main__":
    verify_standalone()
    verify_level_change()
    verify_revision_blend()
    print("Published aggregate metrics: PASS")
