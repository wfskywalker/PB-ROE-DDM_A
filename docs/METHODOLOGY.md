# Methodology

## Research question

Can a company's market price be compared with its point-in-time profitability,
analyst EPS forecasts, required return, and long-run growth assumptions to
produce a cross-sectional expected-return signal? Does the information persist
after removing broad industry, size, and market-beta exposures?

## Universe and timing

The monthly signal is formed after the final A-share trading-day close. Eligible
securities end in `.SH` or `.SZ`; Beijing-listed, ST, financial, and real-estate
stocks are excluded using status and Shenwan membership observable at the
signal date. Annual financial reports must have an announcement date no later
than the signal date. The next trading day's open is used only for simulated
execution, never to form the factor.

Broker reports are deduplicated by stock, normalized broker, and fiscal year,
retaining the latest report published within 180 calendar days before the
signal. A terminal forecast needs at least two distinct brokers. The latest
eligible terminal fiscal year is selected from the signal year and following
two years. This actual terminal horizon can be shorter than three years.

## Valuation and residual signal

Weighted-average ROE is preferred when it is consistently available for the
company's latest three announced annual observations; otherwise ordinary ROE
is used consistently. The steady ROE estimate is their arithmetic mean.
Newly listed companies with fewer than three observations shrink toward a
point-in-time industry peer mean. Nonpositive steady ROE invalidates the
signal. Five ex-ante long-run growth buckets (0–4%) map to Shenwan level 3.

```text
justified_PB = (steady_ROE - g) / (cost_of_equity - g)
justified_PE = justified_PB / steady_ROE
target_price = terminal_consensus_EPS × justified_PE
T = (terminal_fiscal_year_end - signal_date) / 365.25
PBROE_IAR = (target_price / signal_close)^(1/T) - 1
cost_of_equity = risk_free_rate + trailing_beta × ERP
```

The model requires `cost_of_equity - g >= 2%`, positive EPS and price, a
positive target, and finite outputs. Dividends are excluded from the signal.
It also omits the original spreadsheet's ten-year intermediate growth stage
and subjective company-level PE adjustments.

Within each month, winsorize the implied return, log market value, and beta at
1%/99%, then z-score them. The primary residual signal is the OLS residual
from:

```text
z(PBROE_IAR) = intercept + SW1 industry effects
             + b1 × z(log market value) + b2 × z(trailing beta) + residual
```

Higher residuals are ranked as more attractive. The residual is a standardized
relative mispricing score; it is **not itself an annualized percentage return**.

## Research challengers

**R1: level + change.** Rank the residual within month `t` as `L_t`.
For stocks also observed at the preceding scheduled month end, form
`D_t = L_t - L_(t-1)` and rank this change as `M_t`. Set
`R1_t = 0.5 L_t + 0.5 M_t`. The same-stock prior-month requirement defines the
R0/R1 matched sample.

**Analyst EPS revision.** Match stock, normalized broker, and terminal fiscal
year exactly. Compare each broker's latest EPS at `t` with its latest
observable EPS at about `t-90`. Require at least two matched brokers and one
active update. Rank the symmetric EPS change and the breadth of upward versus
downward revisions, combine them 50/50, then combine its monthly rank 50/50
with the PBROE residual rank on the exact common sample.

Other explored variants shrink extreme steady ROE and weight predictions by
forecast dispersion and growth sensitivity. They did not clear the frozen
incremental gates; [results](RESULTS.md) records the outcomes.

## Labels and portfolios

The primary label is adjusted open `t+1` to adjusted open `t+21` (20 trading
days). Monthly ten-group portfolios form after signal close and enter at the
next adjusted open. Group 1 has the lowest score, group 10 the highest. The
portfolio simulator carries blocked sales, respects limit and suspension
constraints, and applies buy/sell commissions and sell-side stamp tax.

Monthly Spearman Rank IC relates stock signal ranks to the future 20-day
label. Annualized ICIR is `sqrt(12) × mean(monthly IC) / sample_std(monthly
IC)`. G10-G1 is the difference of after-cost monthly group returns; the
reported annualized spread is its monthly mean times 12. Monotonicity is
reported using Spearman across ten long-run group means, ordered adjacent
pairs, and within-month group correlations.

For implementation boundaries, data requirements, and limitations, see
[data and reproducibility](DATA_AND_REPRODUCIBILITY.md) and
[limitations](LIMITATIONS.md).
