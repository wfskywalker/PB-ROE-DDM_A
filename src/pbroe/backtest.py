"""Cross-sectional deciles and stateful next-open execution simulation."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


def assign_deciles(
    frame: pd.DataFrame,
    factor_column: str = "pbroe_implied_ann_return",
    group_count: int = 10,
) -> pd.DataFrame:
    """Deterministically assign near-equal groups; highest factor is group 10."""
    if group_count <= 1:
        raise ValueError("group_count must exceed one")
    valid = frame.loc[frame[factor_column].notna()].copy()
    valid.sort_values([factor_column, "ts_code"], inplace=True, kind="mergesort")
    n = len(valid)
    if n < group_count:
        raise ValueError("not enough valid observations for requested groups")
    valid["group"] = (
        np.floor(np.arange(n, dtype=float) * group_count / n).astype(int) + 1
    )
    return valid.sort_values("ts_code").reset_index(drop=True)


def assign_industry_deciles(
    frame: pd.DataFrame,
    factor_column: str,
    industry_column: str = "l3_code",
    group_count: int = 10,
    min_industry_size: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign true near-equal deciles independently inside eligible industries."""
    minimum = group_count if min_industry_size is None else min_industry_size
    if minimum < group_count:
        raise ValueError("min_industry_size cannot be below group_count")
    required = {"ts_code", factor_column, industry_column}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"industry-decile inputs missing columns: {sorted(missing)}")
    if frame.ts_code.duplicated().any():
        raise ValueError("ts_code must be unique within a signal month")
    parts, audit_rows = [], []
    for industry, subset in frame.groupby(industry_column, sort=True, dropna=False):
        valid_count = int(subset[factor_column].notna().sum())
        eligible = pd.notna(industry) and valid_count >= minimum
        audit_rows.append(
            {industry_column: industry, "input_rows": len(subset),
             "valid_rows": valid_count, "eligible": bool(eligible)}
        )
        if eligible:
            grouped = subset.loc[subset[factor_column].notna()].copy()
            grouped.sort_values([factor_column, "ts_code"], inplace=True, kind="mergesort")
            n = len(grouped)
            # Midpoint bins distribute indivisible remainders symmetrically
            # rather than systematically giving the lowest groups more names.
            grouped["group"] = (
                np.floor((np.arange(n, dtype=float) + 0.5) * group_count / n)
                .astype(int)
                .clip(0, group_count - 1)
                + 1
            )
            grouped = grouped.sort_values("ts_code").reset_index(drop=True)
            parts.append(grouped.rename(columns={"group": "industry_group"}))
    assigned = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return assigned, pd.DataFrame(audit_rows)


def adjusted_open_table(
    prices: pd.DataFrame, adjustments: pd.DataFrame, trade_filter: pd.DataFrame
) -> pd.DataFrame:
    required_price = {"ts_code", "trade_date", "open"}
    required_adj = {"ts_code", "trade_date", "adj_factor"}
    required_filter = {"ts_code", "trade_date", "is_limit_up", "is_limit_down", "is_st"}
    if required_price - set(prices) or required_adj - set(adjustments) or required_filter - set(trade_filter):
        raise ValueError("execution inputs are missing required columns")
    x = prices[["ts_code", "trade_date", "open"]].merge(
        adjustments[["ts_code", "trade_date", "adj_factor"]],
        on=["ts_code", "trade_date"],
        validate="one_to_one",
    )
    x = x.merge(
        trade_filter[["ts_code", "trade_date", "is_limit_up", "is_limit_down", "is_st"]],
        on=["ts_code", "trade_date"],
        how="left",
        validate="one_to_one",
    )
    x["adjusted_open"] = pd.to_numeric(x["open"], errors="coerce") * pd.to_numeric(
        x["adj_factor"], errors="coerce"
    )
    for column in ("is_limit_up", "is_limit_down", "is_st"):
        x[column] = x[column].fillna(False).astype(bool)
    x["buyable"] = x.adjusted_open.gt(0) & ~x.is_limit_up & ~x.is_st
    x["sellable"] = x.adjusted_open.gt(0) & ~x.is_limit_down
    return x


@dataclass
class StatefulPortfolio:
    buy_commission: float
    sell_commission: float
    stamp_tax: float
    cash: float = 1.0
    shares: dict[str, float] = field(default_factory=dict)
    last_prices: dict[str, float] = field(default_factory=dict)
    baseline_nav: float | None = None
    previous_signal_date: int | None = None

    def rebalance(
        self,
        signal_date: int | None,
        targets: set[str],
        execution: pd.DataFrame,
    ) -> tuple[dict | None, dict]:
        rows = execution.set_index("ts_code").to_dict("index") if not execution.empty else {}
        for code, row in rows.items():
            price = float(row["adjusted_open"])
            if np.isfinite(price) and price > 0:
                self.last_prices[code] = price
        pre_nav = self.cash + sum(
            quantity * self.last_prices.get(code, 0.0)
            for code, quantity in self.shares.items()
        )
        eligible_targets = {
            code for code in targets if code in rows and rows[code]["adjusted_open"] > 0
        }
        desired = pre_nav / len(eligible_targets) if eligible_targets else 0.0
        bought = sold = costs = 0.0

        for code in list(self.shares):
            price = self.last_prices.get(code, 0.0)
            current = self.shares[code] * price
            target_value = desired if code in eligible_targets else 0.0
            sellable = code in rows and bool(rows[code]["sellable"])
            if current > target_value and sellable:
                trade_value = current - target_value
                fee = trade_value * (self.sell_commission + self.stamp_tax)
                self.shares[code] -= trade_value / price
                if self.shares[code] <= 1e-14:
                    self.shares.pop(code, None)
                self.cash += trade_value - fee
                sold += trade_value
                costs += fee

        deficits = {}
        for code in eligible_targets:
            row = rows[code]
            if not bool(row["buyable"]):
                continue
            current = self.shares.get(code, 0.0) * self.last_prices[code]
            if current < desired:
                deficits[code] = desired - current
        total_deficit = sum(deficits.values())
        capacity = self.cash / (1.0 + self.buy_commission)
        scale = min(1.0, capacity / total_deficit) if total_deficit > 0 else 0.0
        for code, deficit in deficits.items():
            trade_value = deficit * scale
            fee = trade_value * self.buy_commission
            self.shares[code] = self.shares.get(code, 0.0) + trade_value / self.last_prices[code]
            self.cash -= trade_value + fee
            bought += trade_value
            costs += fee

        post_nav = self.cash + sum(
            quantity * self.last_prices.get(code, 0.0)
            for code, quantity in self.shares.items()
        )
        result = None
        if self.previous_signal_date is not None and self.baseline_nav is not None:
            result = {
                "signal_date": self.previous_signal_date,
                "net_return": post_nav / self.baseline_nav - 1.0,
                "ending_nav": post_nav,
            }
        event = {
            "signal_date": signal_date,
            "pretrade_nav": pre_nav,
            "posttrade_nav": post_nav,
            "buy_value": bought,
            "sell_value": sold,
            "cost": costs,
            "turnover": (bought + sold) / pre_nav if pre_nav > 0 else np.nan,
            "holding_count": len(self.shares),
            "blocked_target_buys": len(targets - {c for c in targets if c in rows and rows[c]["buyable"]}),
            "blocked_legacy_sells": sum(
                1
                for code in self.shares
                if code not in targets and not (code in rows and rows[code]["sellable"])
            ),
        }
        if self.baseline_nav is None:
            self.baseline_nav = pre_nav
        else:
            self.baseline_nav = post_nav
        self.previous_signal_date = signal_date
        return result, event
