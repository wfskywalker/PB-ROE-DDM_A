# Limitations and open research questions

1. **No untouched holdout.** The historical sample was inspected during
   development. The reported tests establish historical evidence, not a fresh
   out-of-sample discovery or a live trading result.
2. **Turnover and capacity.** R1 raises mean G10 turnover from 62.0% to
   158.4% per month. Explicit commissions and stamp tax are modeled, but
   market impact, participation rate, and portfolio capacity are not. The
   preregistered turnover gate failed, so R0 remains the governed baseline.
3. **Terminal-horizon instability.** The allowed terminal fiscal year can be
   close to the signal date at year-end. Annualizing over a near-zero horizon
   produces extreme raw values. Monthly winsorization limits their rank
   influence; a separately frozen minimum-horizon rule still needs testing.
4. **Forecast availability.** The signal relies on at least two brokers with
   same-year forecasts. Its coverage is not representative of every A-share.
   Backfilled or erroneous report dates remain a vendor-level point-in-time
   risk.
5. **Model assumptions.** Steady ROE, perpetual-growth buckets, beta, ERP,
   accounting quality, and terminal EPS affect the implied target. Historical
   beta and fixed ex-ante growth policies are reproducible approximations.
6. **Portfolio interpretation.** G10-G1 is an after-cost difference between
   two long-only group returns. Its arithmetic annualized mean is a factor
   spread. A geometric wealth curve of `1 + spread` is only a risk diagnostic,
   not a directly funded long-short account.
7. **ML integration.** The standalone signal and analyst-revision blend have
   not established incremental value in a completed RF walk-forward
   production replay. They were not added to the live RF strategy.

Next gates: minimum-horizon specification, prospective shadow run, capacity
and impact replay, and a frozen walk-forward incremental feature test.
