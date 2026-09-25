"""Historical month-end schedules and trailing beta materialization."""

from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from .universe import month_end_trade_dates


def backtest_signal_schedule(
    trade_calendar: pd.DataFrame, start_date: int, end_date: int
) -> pd.DataFrame:
    """Return month-end signals with next-open entry and following entry dates."""
    opened = trade_calendar.loc[trade_calendar["is_open"].eq(1), ["cal_date"]].copy()
    opened["cal_date"] = pd.to_numeric(opened["cal_date"], errors="raise").astype(int)
    opened.sort_values("cal_date", inplace=True)
    dates = opened["cal_date"].tolist()
    position = {date: index for index, date in enumerate(dates)}
    all_signals = month_end_trade_dates(trade_calendar)
    signals = [
        date
        for date in all_signals
        if int(start_date) <= date <= int(end_date)
    ]
    rows = []
    for index, signal_date in enumerate(signals):
        trade_index = position[signal_date]
        if trade_index + 21 >= len(dates):
            continue
        all_index = all_signals.index(signal_date)
        next_signal = (
            all_signals[all_index + 1] if all_index + 1 < len(all_signals) else None
        )
        next_entry = (
            dates[position[next_signal] + 1]
            if next_signal is not None and position[next_signal] + 1 < len(dates)
            else None
        )
        rows.append(
            {
                "signal_date": signal_date,
                "entry_date": dates[trade_index + 1],
                "label_exit_date": dates[trade_index + 21],
                "next_signal_date": next_signal,
                "next_entry_date": next_entry,
            }
        )
    return pd.DataFrame(rows).astype(
        {
            "signal_date": "int64",
            "entry_date": "int64",
            "label_exit_date": "int64",
            "next_signal_date": "Int64",
            "next_entry_date": "Int64",
        }
    )


def materialize_month_end_betas(
    asset_pricing_root: str | Path,
    signal_dates: list[int],
    output_path: str | Path,
    window_days: int = 252,
    min_observations: int = 120,
) -> Path:
    """Calculate trailing equal-weight-market beta once for all signal dates."""
    if not signal_dates:
        raise ValueError("signal_dates must not be empty")
    if window_days <= 1 or min_observations <= 1 or min_observations > window_days:
        raise ValueError("invalid beta window or minimum observations")
    root = Path(asset_pricing_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pv_glob = str(root / "data/stock_data/daily/pv/*/*.parquet").replace("'", "''")
    adj_glob = str(root / "data/stock_data/daily/adj_factor/*/*.parquet").replace("'", "''")
    values = ",".join(f"({int(value)})" for value in sorted(set(signal_dates)))
    max_date = max(signal_dates)
    sql = f"""
    COPY (
      WITH signal_dates(signal_date) AS (VALUES {values}),
      prices AS (
        SELECT p.trade_date, p.ts_code,
               p.close * CAST(a.adj_factor AS DOUBLE) AS adjusted_close,
               dense_rank() OVER (ORDER BY p.trade_date) AS trade_index
        FROM read_parquet('{pv_glob}', union_by_name=true) p
        JOIN read_parquet('{adj_glob}', union_by_name=true) a
          USING (trade_date, ts_code)
        WHERE p.trade_date <= {int(max_date)}
      ),
      lagged AS (
        SELECT *, adjusted_close /
          lag(adjusted_close) OVER (PARTITION BY ts_code ORDER BY trade_date) - 1.0
          AS stock_return
        FROM prices
      ),
      market AS (
        SELECT trade_date, avg(stock_return) AS market_return
        FROM lagged
        GROUP BY trade_date
      ),
      paired AS (
        SELECT l.*, m.market_return
        FROM lagged l JOIN market m USING (trade_date)
      ),
      rolling AS (
        SELECT trade_date AS signal_date, ts_code,
               count(stock_return) OVER w AS beta_observations,
               min(CASE WHEN stock_return IS NOT NULL THEN trade_date END) OVER w
                 AS beta_start_date,
               max(CASE WHEN stock_return IS NOT NULL THEN trade_date END) OVER w
                 AS beta_end_date,
               covar_samp(stock_return, market_return) OVER w /
                 nullif(
                   var_samp(market_return) FILTER (WHERE stock_return IS NOT NULL) OVER w,
                   0.0
                 ) AS raw_beta
        FROM paired
        WINDOW w AS (
          PARTITION BY ts_code ORDER BY trade_index
          RANGE BETWEEN {int(window_days) - 1} PRECEDING AND CURRENT ROW
        )
      )
      SELECT r.signal_date, r.ts_code,
             CASE WHEN r.beta_observations >= {int(min_observations)}
                  THEN r.raw_beta ELSE NULL END AS beta,
             r.beta_observations, r.beta_start_date, r.beta_end_date
      FROM rolling r JOIN signal_dates s USING (signal_date)
      ORDER BY r.signal_date, r.ts_code
    ) TO '{str(output).replace("'", "''")}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    connection = duckdb.connect()
    try:
        connection.execute("SET threads=4")
        connection.execute(sql)
    finally:
        connection.close()
    return output
