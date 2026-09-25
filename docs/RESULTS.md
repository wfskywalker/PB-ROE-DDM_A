# Research results

All values are historical research results, not live trading returns.
The sample and denominator are as important as the headline number.

## 1. Standalone residual PB-ROE

172 month ends, March 2012–June 2026.

| Signal | Mean Rank IC | Annualized ICIR | Positive IC months | After-cost G10-G1 annualized |
|---|---:|---:|---:|---:|
| Raw implied annualized return | 0.03311 | 0.604 | 57.6% | 2.09% |
| SW1/size/beta residual | **0.03354** | **1.595** | **69.2%** | **7.56%** |

The raw factor failed the frozen time-stability gate; the residual version
passed. Its three fixed subperiod G10-G1 annualized spreads were 10.61%,
5.48%, and 6.58%. See
[`raw_vs_neutralized_summary.csv`](../results/raw_vs_neutralized_summary.csv).

## 2. Residual level plus one-month change

Exact R0/R1 common sample: 200,468 stock-months and 171 month ends,
April 2012–June 2026. This is not the standalone 172-month population.

| Signal | Mean Rank IC | ICIR | G10-G1 annualized | G10-G1 Sharpe | G10 turnover/month |
|---|---:|---:|---:|---:|---:|
| R0 residual level | 0.03306 | 1.538 | 7.36% | 0.697 | 62.03% |
| R1 level + change | **0.04538** | **1.990** | **13.78%** | **1.176** | **158.39%** |

Paired monthly IC improvement: +0.01232 (t=4.08; Holm-adjusted
`p=0.00021`). A six-month moving-block bootstrap gave a 95% interval
of [0.00608, 0.01897] for the IC delta. All three fixed subperiod ICs
improved. The pre-registered advancement gate capped the turnover increase
at 25 percentage points; the actual rise was 96.36 points. Therefore R1
remains a high-turnover research challenger, and R0 stays the governed
baseline. The explicit-fee 5× scenario remained positive but does not
model impact or capacity.

Long-run group-return Spearman was 0.976 for both R0 and R1, with 7 of 9
adjacent decile pairs ordered. Average within-month group Spearman improved
from 0.172 to 0.230. Most spread improvement came from a lower G1, with only
a modest rise in long-only G10. See
[`all_summary.csv`](../results/all_summary.csv),
[`all_monotonicity.csv`](../results/all_monotonicity.csv), and the
[decile chart](figures/decile_profiles.png).

## 3. Independent analyst-revision blend

Separate common sample: 111,682 stock-months over 172 month ends. Its PBROE
baseline is not directly comparable with R0 in section 2.

| Signal | Mean Rank IC | ICIR | G10-G1 annualized | G10-G1 Sharpe |
|---|---:|---:|---:|---:|
| PBROE on revision-eligible sample | 0.03379 | 1.209 | 8.20% | 0.551 |
| 50/50 PBROE + analyst EPS revision | **0.03966** | **1.699** | **15.80%** | **1.093** |

Long-run decile Spearman improved from 0.879 to 0.952 and adjacent ordered
pairs from 6/9 to 7/9. This blend improved both G10 and G1 on its own
matched sample. See
[`common_specification_summary.csv`](../results/common_specification_summary.csv)
and [`common_specification_monotonicity.csv`](../results/common_specification_monotonicity.csv).

## Interpretation

The signal has historical cross-sectional predictive evidence, but its
enhancements have not established capacity-adjusted live value or incremental
performance in a completed RF walk-forward replay. The valuation-level and
analyst-blend comparisons have different populations. Recompute selected
figures from included monthly aggregate series with:

```bash
python examples/verify_published_results.py
```
