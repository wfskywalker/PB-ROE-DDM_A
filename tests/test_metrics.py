import numpy as np
import pandas as pd
import pytest

from pbroe.metrics import monotonicity_summary, performance_summary


def test_performance_summary_matches_independent_calculation():
    returns = pd.Series([0.10, -0.05, 0.03, 0.02] * 3)
    result = performance_summary(returns)
    wealth = (1 + returns).cumprod()
    path = pd.concat([pd.Series([1.0]), wealth], ignore_index=True)
    expected_dd = (path / path.cummax() - 1).min()
    expected_cagr = wealth.iloc[-1] - 1
    assert result["compound_annual_return"] == pytest.approx(expected_cagr)
    assert result["maximum_drawdown"] == pytest.approx(expected_dd)
    assert result["calmar"] == pytest.approx(expected_cagr / abs(expected_dd))
    assert result["sharpe_zero_rf"] == pytest.approx(
        np.sqrt(12) * returns.mean() / returns.std(ddof=1)
    )


def test_monotonicity_summary_detects_perfect_and_imperfect_order():
    perfect = pd.DataFrame({group: [group, group + 1] for group in range(1, 11)})
    result = monotonicity_summary(perfect)
    assert result["full_history_group_spearman"] == pytest.approx(1.0)
    assert result["adjacent_pair_hit_ratio"] == pytest.approx(1.0)
    assert result["mean_monthly_group_spearman"] == pytest.approx(1.0)
    imperfect = perfect.copy()
    imperfect[[9, 10]] = imperfect[[10, 9]].to_numpy()
    result = monotonicity_summary(imperfect)
    assert result["adjacent_pair_hit_ratio"] < 1.0
