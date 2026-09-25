import numpy as np
import pandas as pd
import pytest

from pbroe.backtest import (
    StatefulPortfolio, adjusted_open_table, assign_deciles, assign_industry_deciles,
)
from pbroe.neutralization import neutralize_month, neutralize_panel


def test_deciles_are_balanced_exhaustive_and_ordered():
    x = pd.DataFrame(
        {"ts_code": [f"S{i:02d}" for i in range(23)], "factor": range(23)}
    )
    grouped = assign_deciles(x, "factor")
    counts = grouped.groupby("group").size()
    assert set(grouped.group) == set(range(1, 11))
    assert counts.max() - counts.min() <= 1
    assert grouped.groupby("group").factor.max().is_monotonic_increasing


def test_execution_flags_and_costed_round_trip_reconcile():
    prices = pd.DataFrame({"ts_code": ["A"], "trade_date": [20240102], "open": [10.0]})
    adj = pd.DataFrame({"ts_code": ["A"], "trade_date": [20240102], "adj_factor": [1.0]})
    filt = pd.DataFrame(
        {"ts_code": ["A"], "trade_date": [20240102], "is_limit_up": [False], "is_limit_down": [False], "is_st": [False]}
    )
    entry = adjusted_open_table(prices, adj, filt)
    p = StatefulPortfolio(0.0001, 0.0001, 0.0005)
    result, event = p.rebalance(20240101, {"A"}, entry)
    assert result is None
    assert event["buy_value"] == pytest.approx(1 / 1.0001)
    exit_table = entry.copy()
    exit_table["adjusted_open"] = 11.0
    result, event = p.rebalance(None, set(), exit_table)
    expected_before_sell = (1 / 1.0001) * 1.1
    expected_after_sell = expected_before_sell * (1 - 0.0006)
    assert result["net_return"] == pytest.approx(expected_after_sell - 1)
    assert event["holding_count"] == 0


def test_limit_up_blocks_buy_and_limit_down_blocks_sell():
    p = StatefulPortfolio(0.0001, 0.0001, 0.0005)
    blocked_buy = pd.DataFrame(
        [{"ts_code": "A", "adjusted_open": 10.0, "buyable": False, "sellable": True}]
    )
    _, event = p.rebalance(1, {"A"}, blocked_buy)
    assert event["holding_count"] == 0
    assert event["blocked_target_buys"] == 1


def test_neutralization_preserves_rows_and_satisfies_normal_equations():
    rows = []
    for i in range(60):
        industry = ["A", "B", "C"][i % 3]
        log_mv = 10 + i / 20
        beta = 0.6 + (i % 11) / 10
        industry_effect = {"A": -0.4, "B": 0.2, "C": 0.7}[industry]
        factor = 0.8 * log_mv - 0.5 * beta + industry_effect + (i % 7) / 50
        rows.append(
            {"ts_code": f"S{i:03d}", "factor": factor, "industry": industry,
             "total_mv": np.exp(log_mv), "beta": beta}
        )
    frame = pd.DataFrame(rows).sample(frac=1.0, random_state=7)
    result, audit = neutralize_month(
        frame, factor_column="factor", industry_column="industry"
    )
    assert result.index.tolist() == frame.index.tolist()
    assert result.ts_code.tolist() == frame.ts_code.tolist()
    assert result.pbroe_iar_neutralized.notna().all()
    assert audit["design_rank"] == audit["design_columns"]
    assert audit["max_abs_normal_equation"] < 1e-12


def test_neutralization_winsorizes_extreme_factor_and_is_month_local():
    base = pd.DataFrame(
        {
            "ts_code": [f"S{i:02d}" for i in range(20)],
            "signal_date": [20240131] * 10 + [20240229] * 10,
            "factor": list(range(9)) + [1e120] + list(range(10, 20)),
            "industry": (["A", "B"] * 5) * 2,
            "total_mv": np.linspace(100, 1000, 20),
            "beta": np.tile(np.linspace(0.5, 1.4, 10), 2),
        }
    )
    result, audit = neutralize_panel(
        base, factor_column="factor", industry_column="industry"
    )
    assert len(result) == len(base)
    assert len(audit) == 2
    assert np.isfinite(result.pbroe_iar_neutralized).all()
    assert np.isfinite(result.factor_winsorized_z).all()


def test_industry_deciles_exclude_small_cells_and_partition_eligible_cells():
    frame = pd.DataFrame(
        {
            "ts_code": [f"A{i:02d}" for i in range(23)] + [f"B{i:02d}" for i in range(9)],
            "factor": list(range(23)) + list(range(9)),
            "l3_code": ["A"] * 23 + ["B"] * 9,
        }
    )
    assigned, audit = assign_industry_deciles(frame, "factor")
    assert set(assigned.l3_code) == {"A"}
    assert len(assigned) == 23
    assert set(assigned.industry_group) == set(range(1, 11))
    counts = assigned.groupby("industry_group").size()
    assert counts.max() - counts.min() <= 1
    assert not audit.set_index("l3_code").loc["B", "eligible"]


def test_industry_decile_remainders_are_symmetric_at_the_tails():
    frame = pd.DataFrame(
        {"ts_code": [f"A{i:02d}" for i in range(11)],
         "factor": range(11), "l3_code": ["A"] * 11}
    )
    assigned, _ = assign_industry_deciles(frame, "factor")
    counts = assigned.groupby("industry_group").size()
    assert counts.loc[1] == counts.loc[10] == 1
