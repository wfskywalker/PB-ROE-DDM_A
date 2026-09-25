import numpy as np
import pandas as pd
import pytest

from pbroe.refinement import (
    add_confidence_score,
    add_level_change_score,
    leave_one_out_peer_clip,
)


def test_level_change_requires_immediately_prior_scheduled_month():
    panel = pd.DataFrame({
        "signal_date": [20240131, 20240131, 20240229, 20240229, 20240329],
        "ts_code": ["A", "B", "A", "C", "A"],
        "factor": [1.0, 2.0, 2.0, 1.0, 3.0],
    })
    out = add_level_change_score(panel, "factor", "r1")
    assert out.loc[(out.signal_date == 20240229) & (out.ts_code == "C"), "r1"].isna().all()
    assert out.loc[(out.signal_date == 20240329) & (out.ts_code == "A"), "r1"].notna().all()
    assert out.loc[out.signal_date == 20240131, "r1"].isna().all()


def test_level_change_is_fixed_equal_weight_formula():
    panel = pd.DataFrame({
        "signal_date": [20240131] * 2 + [20240229] * 2,
        "ts_code": ["A", "B", "A", "B"],
        "factor": [1.0, 2.0, 2.0, 1.0],
    })
    out = add_level_change_score(panel, "factor", "r1")
    feb = out.loc[out.signal_date.eq(20240229)].set_index("ts_code")
    assert feb.loc["A", "r1"] == pytest.approx(1.0)
    assert feb.loc["B", "r1"] == pytest.approx(0.5)


def test_peer_clip_uses_leave_one_out_arithmetic_statistics_and_ceiling():
    values = [0.10] * 11 + [2.0]
    frame = pd.DataFrame({
        "ts_code": [f"S{i}" for i in range(12)],
        "steady_roe": values,
        "l3_code": ["L3"] * 12,
        "l2_code": ["L2"] * 12,
        "l1_code": ["L1"] * 12,
    })
    out = leave_one_out_peer_clip(frame)
    extreme = out.loc[out.ts_code.eq("S11")].iloc[0]
    assert extreme.peer_level_roe == "l3_code"
    assert extreme.peer_count_roe == 11
    assert extreme.peer_mean_roe == pytest.approx(0.10)
    assert extreme.steady_roe_r2 <= 0.50
    assert bool(extreme.roe_was_transformed)
    assert np.isfinite(out.steady_roe_r2).all()


def test_confidence_shrinks_toward_neutral_and_never_reverses_sign():
    panel = pd.DataFrame({
        "signal_date": [20240131] * 3,
        "r2_score": [0.9, 0.1, 0.8],
        "eps_dispersion": [0.1, 0.2, np.nan],
        "growth_sensitivity": [0.1, np.nan, 0.3],
    })
    out = add_confidence_score(panel)
    assert out.r3_confidence.between(0, 1).all()
    assert ((out.r3_score - 0.5) * (out.r2_score - 0.5) >= 0).all()
    assert (out.r3_score - 0.5).abs().le((out.r2_score - 0.5).abs()).all()
    assert out.loc[2, "r3_confidence"] == 0.0
