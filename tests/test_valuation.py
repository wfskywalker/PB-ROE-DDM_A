import pytest

from pbroe.valuation import value_pbroe


def test_golden_calculation_and_actual_horizon():
    result = value_pbroe(20250630, 2027, 20, 2, 0.20, 0.10, 0.03)
    pb = (0.20 - 0.03) / (0.10 - 0.03)
    pe = pb / 0.20
    target = 2 * pe
    horizon = (pytest.importorskip("pandas").Timestamp("2027-12-31") - pytest.importorskip("pandas").Timestamp("2025-06-30")).days / 365.25
    assert result.justified_pb == pytest.approx(pb)
    assert result.target_price == pytest.approx(target)
    assert result.horizon_years == pytest.approx(horizon)
    assert result.implied_ann_return == pytest.approx((target / 20) ** (1 / horizon) - 1)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"close": 0}, "NON_POSITIVE_PRICE"),
        ({"eps_fy3": 0}, "NON_POSITIVE_EPS"),
        ({"steady_roe": 0}, "NON_POSITIVE_STEADY_ROE"),
        ({"cost_of_equity": 0.049, "perpetual_growth": 0.03}, "COE_G_SPREAD_TOO_SMALL"),
        ({"signal_date": 20271231, "fy3_year": 2026}, "NON_POSITIVE_HORIZON"),
    ],
)
def test_domain_guards(kwargs, reason):
    base = dict(signal_date=20250630, fy3_year=2027, close=20, eps_fy3=2, steady_roe=0.20, cost_of_equity=0.10, perpetual_growth=0.03)
    base.update(kwargs)
    assert value_pbroe(**base).reason == reason


def test_economic_directions():
    base = dict(signal_date=20250630, fy3_year=2027, close=20, eps_fy3=2, steady_roe=0.20, cost_of_equity=0.10, perpetual_growth=0.03)
    value = value_pbroe(**base).implied_ann_return
    assert value_pbroe(**{**base, "close": 18}).implied_ann_return > value
    assert value_pbroe(**{**base, "eps_fy3": 2.2}).implied_ann_return > value
    assert value_pbroe(**{**base, "steady_roe": 0.22}).implied_ann_return > value
    assert value_pbroe(**{**base, "cost_of_equity": 0.11}).implied_ann_return < value
    assert value_pbroe(**{**base, "perpetual_growth": 0.04}).implied_ann_return > value
