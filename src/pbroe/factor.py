"""Join validated point-in-time inputs into PBROE_IAR."""

import numpy as np
import pandas as pd

from .valuation import value_pbroe


def generate_factor_snapshot(
    universe, consensus, roe, beta, prices, growth_mapping, signal_date,
    risk_free_rate, equity_risk_premium, min_coe_growth_spread=0.02,
):
    base = universe.copy()
    keep_consensus = [c for c in consensus.columns if c == "ts_code" or c.startswith(("eps_", "broker_count_", "latest_report_", "forecast_year_", "consensus_", "terminal_"))]
    keep_roe = ["ts_code", "steady_roe", "roe_metric", "roe_observation_count", "latest_roe_ann_date", "industry_prior_level", "industry_peer_count", "roe_reason", "roe_valid"]
    x = base.merge(consensus[keep_consensus], on="ts_code", how="left", validate="one_to_one")
    x = x.merge(roe[[c for c in keep_roe if c in roe]], on="ts_code", how="left", validate="one_to_one")
    x = x.merge(beta, on="ts_code", how="left", validate="one_to_one")
    x = x.merge(prices[["ts_code", "trade_date", "close"]], on="ts_code", how="left", validate="one_to_one")
    x = x.merge(growth_mapping[["l3_code", "perpetual_growth", "policy_version", "effective_from"]], on="l3_code", how="left", validate="many_to_one")
    x["risk_free_rate"] = risk_free_rate
    x["equity_risk_premium"] = equity_risk_premium
    x["cost_of_equity"] = risk_free_rate + x["beta"] * equity_risk_premium

    outputs = []
    for row in x.itertuples(index=False):
        consensus_valid = getattr(row, "consensus_valid", False)
        if pd.isna(consensus_valid) or not bool(consensus_valid):
            outputs.append((np.nan, "INVALID_CONSENSUS", np.nan, np.nan, np.nan, np.nan))
            continue
        roe_valid = getattr(row, "roe_valid", False)
        if pd.isna(roe_valid) or not bool(roe_valid):
            outputs.append((np.nan, getattr(row, "roe_reason", "INVALID_ROE"), np.nan, np.nan, np.nan, np.nan))
            continue
        if not np.isfinite(getattr(row, "beta", np.nan)):
            outputs.append((np.nan, "MISSING_BETA", np.nan, np.nan, np.nan, np.nan))
            continue
        result = value_pbroe(
            signal_date, int(row.terminal_forecast_year), row.close, row.terminal_eps,
            row.steady_roe, row.cost_of_equity, row.perpetual_growth,
            min_coe_growth_spread,
        )
        outputs.append((result.implied_ann_return, result.reason, result.target_price, result.justified_pb, result.justified_pe, result.horizon_years))
    x[["pbroe_implied_ann_return", "reason_code", "target_price", "justified_pb", "justified_pe", "horizon_years"]] = pd.DataFrame(outputs, index=x.index)
    x["factor_valid"] = x.reason_code.eq("VALID")
    x["signal_date"] = int(signal_date)
    return x.sort_values("ts_code").reset_index(drop=True)
