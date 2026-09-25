# Data and reproducibility

## What is public here

The repository includes the factor construction, point-in-time consensus and
ROE logic, regression neutralization, analyst-revision functions, stateful
portfolio simulator, unit tests, a wholly synthetic example, and aggregated
research outputs in `results/`. Monthly IC and decile portfolio series contain
no stock identifiers, raw analyst reports, raw prices, or vendor records.

Run without market-data access:

```bash
python -m pip install -e ".[dev]"
python -m pytest
python examples/synthetic_demo.py
python examples/verify_published_results.py
```

The verifier recalculates key published metrics from the included monthly
aggregate series. It does not regenerate the stock-level signals or prove
point-in-time vendor history. The unit tests check formula and temporal
contracts on synthetic frames.

## What a full historical rerun needs

The 2012–2026 A-share study used separately licensed historical market,
financial, industry, ST status, and analyst-report data. These inputs are not
redistributed. A full independent rerun needs:

1. Exchange trading calendar and month-end signal dates.
2. Point-in-time stock listing, ST status, and Shenwan industry memberships.
3. Report-level broker EPS forecasts with publication/report dates and broker
   identities, including forecast fiscal year.
4. Annual ROE/weighted-average ROE with announcement dates.
5. Daily adjusted open/close, adjustment factors, tradability/limit states,
   market returns, total market value, and a versioned risk-free/ERP policy.

For each month, construct the eligible universe and forecasts as of the
month-end close; calculate PBROE implied return; neutralize against SW1,
log-market-cap and trailing beta; rank ten equal-count groups; form at the
next open; then measure the 20-trading-day open-to-open Rank IC and the
next-rebalance-open portfolio return. See
[methodology](METHODOLOGY.md) for precise definitions.

The published tables are research snapshots. Data-vendor revisions,
industry-history versions, corporate-action adjustments, and execution
assumptions can change a rerun. The original Excel/PPT and sell-side PDFs
are excluded because this repository does not grant redistribution rights.

## Scope of the three headline comparisons

| Comparison | Periods | Sample | Purpose |
|---|---:|---|---|
| Raw vs residual PBROE | 172 months | All valid PBROE signals | Standalone factor |
| R0 vs level/change R1 | 171 months | Stocks with a prior monthly residual signal | Change enhancement |
| PBROE vs analyst-revision blend | 172 months | Stocks with both signals (111,682 stock-months) | Orthogonal signal combination |

Do not combine the most favorable statistics from these different samples into
one claimed strategy result.
