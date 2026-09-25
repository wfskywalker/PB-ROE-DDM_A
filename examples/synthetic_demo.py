"""End-to-end PB-ROE signal demonstration using synthetic, nonmarket data."""

from __future__ import annotations

import numpy as np
import pandas as pd

from pbroe.consensus import build_consensus_asof
from pbroe.factor import generate_factor_snapshot
from pbroe.neutralization import neutralize_month
from pbroe.revisions import build_revision_components, form_revision_score


SIGNAL_DATE = 20250630
INDUSTRIES = [
    ("Electronics", "SIM-ELEC", 0.04),
    ("Healthcare", "SIM-HEALTH", 0.03),
    ("Auto", "SIM-AUTO", 0.02),
]


def make_synthetic_inputs() -> tuple[pd.DataFrame, ...]:
    stocks = []
    reports = []
    roe = []
    beta = []
    prices = []
    for i in range(18):
        code = f"SIM{i + 1:03d}.{'SH' if i % 2 == 0 else 'SZ'}"
        l1_name, l3_code, _ = INDUSTRIES[i % len(INDUSTRIES)]
        stocks.append({
            "ts_code": code,
            "l1_name": l1_name,
            "l3_code": l3_code,
            "total_mv": 80.0 + 13.0 * i + 5.0 * (i % 3),
        })
        roe.append({"ts_code": code, "steady_roe": 0.13 + 0.009 * (i % 7),
                    "roe_valid": True, "roe_reason": "VALID"})
        beta.append({"ts_code": code, "beta": 0.85 + 0.12 * (i % 6)})
        prices.append({"ts_code": code, "trade_date": SIGNAL_DATE,
                       "close": 14.0 + 0.63 * i + 0.25 * (i % 4)})
        for broker in range(3):
            baseline = 1.9 + 0.07 * i + 0.035 * broker
            revision = 0.025 * ((i % 5) - 2) + 0.005 * broker
            for report_date, eps in (
                (20250315, baseline),
                (20250612, baseline * (1.0 + revision)),
            ):
                reports.append({
                    "ts_code": code,
                    "report_date": report_date,
                    "org_name": f"Synthetic Broker {broker + 1}",
                    "quarter": "2027 FY",
                    "eps": eps,
                })
    growth = pd.DataFrame([
        {"l3_code": code, "perpetual_growth": rate,
         "policy_version": "synthetic-v1", "effective_from": 20250101}
        for _, code, rate in INDUSTRIES
    ])
    return (pd.DataFrame(stocks), pd.DataFrame(reports), pd.DataFrame(roe),
            pd.DataFrame(beta), pd.DataFrame(prices), growth)


def main() -> None:
    universe, reports, roe, beta, prices, growth = make_synthetic_inputs()
    consensus = build_consensus_asof(reports, SIGNAL_DATE)
    assert len(consensus.eligible) == len(universe)
    assert consensus.audit.terminal_broker_count.ge(2).all()

    snapshot = generate_factor_snapshot(
        universe, consensus.audit, roe, beta, prices, growth,
        signal_date=SIGNAL_DATE,
        risk_free_rate=0.017,
        equity_risk_premium=0.055,
    )
    valid = snapshot.loc[snapshot.factor_valid].copy()
    assert len(valid) == len(universe)
    neutralized, diagnostics = neutralize_month(valid)
    assert diagnostics["max_abs_normal_equation"] < 1e-10

    targets = consensus.eligible[
        ["ts_code", "terminal_forecast_year"]
    ].rename(columns={"terminal_forecast_year": "forecast_year"})
    targets.insert(0, "signal_date", SIGNAL_DATE)
    _, revision_components = build_revision_components(reports, targets)
    revisions = form_revision_score(revision_components)
    assert len(revisions) == len(universe)

    demo = neutralized[[
        "ts_code", "pbroe_implied_ann_return", "pbroe_iar_neutralized"
    ]].merge(
        revisions[["ts_code", "analyst_revision_score"]],
        on="ts_code", validate="one_to_one"
    )
    demo["pbroe_rank"] = demo.pbroe_iar_neutralized.rank(pct=True)
    demo["combined_rank"] = 0.5 * (
        demo.pbroe_rank + demo.analyst_revision_score.rank(pct=True)
    )
    assert np.isfinite(demo.select_dtypes("number")).all().all()
    print("SYNTHETIC EXAMPLE — no real securities or forecasts")
    print(f"Signal date: {SIGNAL_DATE}; stocks: {len(demo)}; brokers per stock: 3")
    print(f"Neutralization normal-equation error: {diagnostics['max_abs_normal_equation']:.2e}")
    print(demo.sort_values("combined_rank", ascending=False).head(5).to_string(
        index=False, float_format=lambda value: f"{value:.4f}"
    ))


if __name__ == "__main__":
    main()
