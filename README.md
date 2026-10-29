# attribution

Return attribution for Indian equity portfolios held by institutions (PMS, AIF,
family offices). Two frameworks will be built on this foundation:

- **Decision view** — macro/exposure, sector allocation, security selection, interaction
- **Factor view** — market, size, value, momentum, quality, low volatility, plus specific return

Both must explain the same total active return. **This release contains the
contracts only**: input schemas and loaders, a synthetic sample generator, the
output result schema, and tests. There is no attribution maths yet.

```
attribution/
  schemas.py   input contracts, enums, Policy, validation errors
  io.py        CSV / Parquet / JSON loaders and writers that validate
  sample.py    seeded synthetic sample portfolio (python -m attribution.sample)
  result.py    AttributionResult: the output the app consumes, JSON in/out
tests/         pytest suite
```

## Setup

Python 3.11+.

```bash
pip install -e ".[dev]"
pytest
python -m attribution.sample --out sample_data            # CSV
python -m attribution.sample --out sample_data --format parquet --end 2026-09-30 --seed 20261028
```

---

## Input contracts

General rules, enforced by `TableSchema.validate` and every loader:

| Rule | Detail |
|---|---|
| Exact columns | Missing, extra or duplicated columns are errors. |
| Dates | Calendar dates, tz-naive, no time of day. Strings must be ISO `YYYY-MM-DD`. |
| Returns | Simple (not log) decimals: `0.012` is 1.2%. Must be finite and `> -1`. A column whose median absolute value exceeds 0.2 is rejected as probably being in percent. Strings like `"1.2%"` are rejected. |
| Weights | Finite, within `[0, 1]` (long-only). Where stated, sum to 1.0 per date within `1e-6`. |
| Sectors | Exact match to one of the 15 names below. No case-folding or trimming. |
| Text / ids | Non-empty, no leading or trailing whitespace. |
| Keys | Each table's key is unique. |
| No coercion | Anything not already what the contract asks for raises `SchemaValidationError`. Every issue found is reported together, each naming the source, column and rows (CSV: physical `line N`; Parquet: `row N`). |

**Timing convention.** A row dated `t` describes the period that *ends* on `t`.
Weights dated `t` are the weights held through that period (set at the close
of the previous one); returns dated `t` are earned over it. Weights and returns
therefore join on the same date with no shifting.

### Sectors (fixed enum, `schemas.Sector`)

Auto · Bankex · Basic Materials · Capital Goods · Consumer Durables · Energy ·
Financial Services · FMCG · Healthcare · Information Technology · Metal ·
Oil & Gas · Power · Realty · Telecommunication

Changing this list is a contract change: bump `result.SCHEMA_VERSION`.

### Tables

| Table | Columns | Key | Extra rules |
|---|---|---|---|
| `Holdings` | date, security_id, weight, sector | date, security_id | weights sum to 1 per date — this is the **equity sleeve**, not the whole portfolio |
| `SecurityMaster` | security_id, name, sector, isin | security_id | ISIN is `IN` + 9 alphanumerics + valid ISO 6166 check digit |
| `Prices` | date, security_id, total_return | date, security_id | |
| `Benchmark` | date, sector, weight, total_return | date, sector | weights sum to 1 per date |
| `PortfolioReturn` | date, total_return | date | whole portfolio, **net of fees** |
| `FactorReturns` | date, factor, total_return | date, factor | factor ∈ market, size, value, momentum, quality, low_volatility |
| `CashReturn` | date, total_return | date | |
| `RegimeSeries` | date, raw_state, confirmed_state, equity_exposure | date | states ∈ normal, elevated, crisis; exposure in `[0, 1]`; may be daily |

`check_holdings_against_master(holdings, master)` cross-checks that every held
security exists and that its sector matches the master.

### Policy

```python
Policy(
    neutral_weights: dict[str, float],          # all 15 sectors, plus optional "Cash"; sum to 1
    bands: dict[str, tuple[float, float]],      # same keys; 0 <= low <= neutral <= high <= 1
    derisk_mode: "cash" | "within_equity",      # "cash" requires a "Cash" key
    review_date: date,
    benchmark_name: str,
)
```

Weights are fractions of the **total** portfolio. Policy equity exposure is
`1 - neutral_weights["Cash"]`. Stored as JSON; a top-level `"_comment"` string
is allowed and ignored.

---

## Output contract: `AttributionResult`

```
meta            portfolio_id, portfolio_name, benchmark_name, currency, frequency,
                linking, engine_version, generated_at, synthetic_data, schema_version
period          start, end, n_periods
totals          portfolio, benchmark, active, ci{level, lower, upper, method}
decision_view   layers[macro, allocation, selection, interaction]   (exactly, in order)
                by_sector[sector, portfolio_weight, benchmark_weight,
                          portfolio_return, benchmark_return,
                          allocation, selection, interaction]
                macro_anatomy{average_equity_exposure, policy_equity_exposure,
                              cash_return, components{name: value}}
factor_view     factors[factor, exposure, factor_return, contribution, t_stat?]
                                                    (exactly the six, in order)
                specific_return{value, t_stat?, information_ratio?}
                diagnostics{method, n_obs, r_squared, tracking_error,
                            residual_volatility, warnings[]}
reconciliation  decision_total, factor_total, difference, tolerance, passes
limitations[]
```

`to_json()` / `from_json()` round-trip exactly. NaN and infinity are refused in
both directions; unknown or missing keys and wrong types are refused on parse.
Construction enforces the bookkeeping below to `1e-9`, so an inconsistent
result cannot be built.

---

## The reconciliation identity

Every later session must produce results that satisfy this. Notation over the
analysis period, all simple returns, cumulative:

```
R_P   portfolio return, net of fees          (from PortfolioReturn, compounded)
R_B   benchmark return                       (sum_s w_B,s * r_B,s per period, compounded)
A     active return  = R_P - R_B
```

**Decision view**

```
A = Macro + Σ_s Allocation_s + Σ_s Selection_s + Σ_s Interaction_s
```

- `Allocation`, `Selection`, `Interaction` are computed on the equity sleeve
  against the benchmark sector weights and returns, then multi-period linked
  using the method named in `meta.linking`.
- `Macro` is everything that comes from not being 100% in the equity sleeve,
  and every residual between the holdings-based gross return and the reported
  net return. It must be itemised in `macro_anatomy.components`, at minimum
  `exposure` (equity vs cash relative to policy), `fees`, and `residual`
  (intra-period trading, pricing gaps), and those components must sum to the
  macro layer. Nothing may be dropped to make the view balance.

**Factor view**

```
A = Σ_k Contribution_k + Specific
```

`Specific` is what the six factors do not explain, so fees and trading residual
land here.

**Reconciliation**

```
decision_total = Σ layers
factor_total   = Σ factor contributions + specific
difference     = decision_total - factor_total
passes         = |difference| <= tolerance  AND  |decision_total - A| <= tolerance
```

Default `tolerance` is `1e-5` (0.1 bp). A result with `passes = False` can be
built and shown, so the app can display a break, but an engine run that does
not pass is a bug.

Enforced by `result.py` today: active = portfolio − benchmark; layer names and
order; macro components (when given) sum to the macro layer; `by_sector` sums equal the allocation, selection and interaction
layers; the six factors in order; `decision_total`, `factor_total`,
`difference` and `passes` all equal what the numbers imply.

---

## Sample data

`attribution.sample` writes a seeded, reproducible synthetic world. Every file
carries a header saying it is synthetic (CSV `#` lines, Parquet metadata
`attribution.comment`, JSON `_comment`).

- 40 securities over all 15 sectors; synthetic ISINs with the `INEZZ` prefix
- 60 month-ends ending at the month end on or before `--end` (default 2026-09-30)
- returns from a known factor model; size loading = −0.6 × standardised log cap
  (correlation with log cap is −1; judge it on log cap, since a few giants
  dominate any arithmetic-mean cap)
- cap-weighted benchmark at sector level
- portfolio holds the 32 smallest names (8 largest excluded), weighted by
  `cap^-0.75` with fixed sector tilts, rebalanced quarterly → persistent
  small/mid-cap tilt (weighted size loading ≈ +0.46, universe average 0)
- size factor earns a deterministic 1.3%/month premium **after** the crisis
  (the small/mid-cap recovery); nothing up to and including the crisis changes,
  so active return is ≈ +4.3pp a year with size explaining ≈ 63% of it
- `tilt_diagnostics(data)` returns these measures; the tests pin them
- equity exposure from the regime's confirmed state (normal 1.00, elevated
  0.85, crisis 0.60); the rest earns cash; 2% p.a. fee deducted
- cash at a T-bill-like 5.5–7.5% a year
- business-daily regime series with one 60-day crisis, short elevated runs
  either side, and raw false alarms the 5-day confirmation filter removes
- `sample_characteristics.csv` holds the true loadings, so the factor view can
  later be checked against what was put in. It is not an input contract.

## Known limits of this release

- Long-only: weights must be in `[0, 1]`. Short books (Category III AIFs)
  need a contract change.
- Single currency (INR); no FX.
- Holdings are the equity sleeve only; cash enters through exposure and
  `CashReturn`, not as a holding.
