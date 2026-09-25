# Public-repository packaging acceptance log

Date: 2026-09-25. This log covers the presentation copy in this repository,
not a rerun of the licensed stock-level historical pipeline.

## Gate 1 — Source inventory: PASS

**Criterion:** identify code, tests, methods, aggregate evidence, raw data,
third-party documents, large generated assets, and absolute-path dependencies
before copying.

**Observed:** original working directory was approximately 3.2 GB. Its raw
vendor downloads, stock-level Parquet panels, sell-side PDFs, spreadsheet,
PowerPoint, website dependencies, and local configuration were excluded.
The public copy contains source modules, tests, versioned assumption tables,
synthetic example, and aggregate result series.

## Gate 2 — Public-content boundary: PASS

**Criterion:** no vendor row-level data, real stock identifiers in results,
credentials, local user path, bundled third-party PDFs, or large binaries;
all documentation links resolve locally.

**Observed:** targeted path/credential scan returned no hits. Result CSVs
contain only monthly IC, decile portfolio or summary statistics. No PDF,
spreadsheet, PowerPoint, Parquet, `.env`, or `config.toml` was selected.
The only binary included is the chart generated from aggregate data.
Documentation relative links resolved.

## Gate 3 — Executability and numerical evidence: PASS

**Criterion:** all packaged unit tests pass; a network-free synthetic example
executes end to end; independently aggregate at least the headline IC, ICIR,
spread, Sharpe and decile monotonicity; inspect the generated chart.

**Observed:** 70 unit tests passed. The synthetic 18-stock, three-broker
example produced valid valuations, residuals and analyst-revision scores;
the residual normal-equation error was below `1e-10`. The aggregate-results
verifier reconciled standalone, R0/R1, and analyst-blend metrics with the
published CSVs. The two-panel decile figure was visually inspected.

## Gate 4 — Local Git and destination: PASS

**Criterion:** repository tracks only reviewed files, has a clean initial
commit, is copied to the exact requested sibling path, and can be opened
there. Remote publication is reported separately from local readiness.

**Observed:** the curated copy was copied to
`QuantProjects/PB-ROE-DDM_AShare`, and a recursive comparison found no
differences between the reviewed staging copy and the requested destination.
The destination is a standalone Git repository on `main`. The initial commit
and final clean-tree check are recorded by the repository history and final
handoff; GitHub remote publication is a separate gate.

## Gate 5 — GitHub publication: PENDING

**Criterion:** the reviewed initial commit is pushed to a public GitHub
repository under the intended account, and the remote repository is opened
to confirm the README and file boundary.

**Observed:** no GitHub repository has yet been created or verified. This
status must not be interpreted as a published repository.
