# attribution

**What it does.** `attribution` explains where an Indian equity portfolio's
return came from, relative to its benchmark, over a period of at least two
years. Give it monthly holdings, security returns, benchmark sector weights
and returns, the portfolio's reported (net) returns, cash returns, factor
returns, a daily regime series and the client's policy. It returns one
validated result that answers two questions about the same active return:
*which decisions produced it* (the decision view) and *which market factors
it rode* (the factor view). Both views add up to the same total, and the
engine refuses to publish a result in which they do not.

> The sample data in `sample_data/` is **synthetic**. It describes no real
> portfolio, and its benchmark weights are indicative only.

---

## The two views

### Decision view: four decisions, then costs

```
  macro (exposure timing)        how much equity vs cash, against the policy weight
+ allocation                     sector over/underweights
+ selection                      stock picking within sectors
+ interaction                    over/underweighting sectors where stocks were picked well or badly
= GROSS active return            portfolio from holdings x prices, vs the policy benchmark
+ fees and trading costs         net (reported) minus gross
= NET active return              reported portfolio vs the policy benchmark
```

* The **macro layer contains only exposure timing**: the effect of moving
  equity away from the static policy weight set by the mandate. It never
  contains fees.
* Allocation, selection and interaction use **Brinson-Fachler**: each sector
  is judged against the *total* benchmark return, so an overweight scores only
  if the sector beat the benchmark.
* "Selection" is a measured stock-picking effect. It is **not** a leftover.

### Factor view: six factors plus specific return

```
  market (active beta)
+ size + value + momentum + quality + low volatility
+ specific return                 what the six factors do not explain
= NET active return
```

A monthly regression of net active return on the factors, with factors
orthogonalised in the fixed order market, size, value, momentum, quality,
low volatility. Each factor's line is **incremental to the ones before it**:
change the order and the split changes, but the total does not.

### Which return series each view uses

| Line | Built on | Linked against |
|---|---|---|
| Four decision layers | GROSS portfolio (holdings x prices, cash at the equity weight) | policy benchmark |
| Fees and trading costs | NET minus GROSS cumulative return | - |
| Factor view | NET portfolio (reported) | policy benchmark |

The **policy benchmark** is `w_pol x benchmark + (1 - w_pol) x cash`, where
`w_pol` is the policy's neutral equity weight. With a 100% equity policy it is
simply the benchmark.

Monthly effects do not add up across months, because returns compound. Every
line is therefore **linked** (Carino by default, Menchero optional), so the
cumulative lines add up exactly to the cumulative active return.

### Lines across the two views are not supposed to match

Only the **totals** of the two views reconcile. In particular, the factor
view's market line and the decision view's macro layer can differ, even in
sign. The macro layer counts only the periods in which exposure was
deliberately moved. The market coefficient is an average active beta over the
whole window, which also picks up the holdings' structural beta. A small and
mid-cap book carries a beta different from one even when exposure is never
changed.

---

## Running it

### From Python (in a notebook cell)

```python
from attribution.engine import EngineConfig, run_attribution
from attribution.cli import write_result

result = run_attribution("/content/attribution_engine/sample_data",
                         config=EngineConfig(portfolio_name="Sample PMS"))
write_result(result, "/content/drive/MyDrive/attribution_outputs", "Sample PMS")
```

`run_attribution(data_dir, policy=None, period=None, config=None)`:

* `policy`: a `Policy`, a path to a policy JSON, or `None` to read `data_dir/policy.json`
* `period`: optional first and last month, e.g. `("2022-01", "2026-09")`
* `config`: `EngineConfig(portfolio_name=..., linking="carino" or "menchero", bootstrap=True, ...)`

### From the command line (in a notebook cell, prefix with `!`)

```
cd /content/attribution_engine && python -m attribution run --data-dir sample_data --out-dir /content/drive/MyDrive/attribution_outputs --portfolio-name "Sample PMS"
```

Options: `--policy FILE`, `--start 2022-01 --end 2026-09`, `--linking menchero`,
`--no-bootstrap` (faster; skips confidence intervals).

Output: `attribution_<portfolio>_<last month-end>.json` plus a `.log` file
beside it, recording the inputs and their SHA-256 hashes, the package version,
the linking method, the factor order, the validation verdict and the UTC time.
**An existing file is never overwritten**: a second run for the same date
writes `..._2.json`, then `_3`, and so on.

To regenerate the synthetic sample: `python -m attribution.sample --out sample_data`.

---

## Inputs

All files are CSV or Parquet, named exactly as below, in one folder. Columns
must match exactly.

**Returns are simple decimal returns: `0.012` means +1.2%.** They are not
percentages and not log returns. Weights are fractions: `0.25` means 25%.
Dates are `YYYY-MM-DD`, with no time and no timezone. A row dated `t`
describes the period ending on `t`: weights dated `t` were held through that
month, and returns dated `t` were earned in it.

| File | Columns | Notes |
|---|---|---|
| `holdings` | date, security_id, weight, sector | equity sleeve only; weights sum to 1 each month |
| `prices` | date, security_id, total_return | monthly total return per security |
| `benchmark` | date, sector, weight, total_return | sector weights sum to 1 each month |
| `portfolio_returns` | date, total_return | reported, **net of fees** |
| `cash_returns` | date, total_return | T-bill or liquid-fund return |
| `factor_returns` | date, factor, total_return | factors: market, size, value, momentum, quality, low_volatility |
| `regime_series` | date, raw_state, confirmed_state, equity_exposure | daily; states `normal` / `elevated` / `crisis` (or 0 / 1 / 2); exposure 0 to 1 |
| `policy.json` | neutral_weights, bands, derisk_mode, review_date, benchmark_name | neutral weights and bands for each sector plus `Cash`; derisk_mode `cash` or `within_equity` |
| `security_master` (optional) | security_id, name, sector, isin | Indian ISINs with a valid check digit |

**Sectors.** Use exactly these 13 names, a condensed approximation of the NSE
sector classification: Auto, Basic Materials, Capital Goods, Consumer
Durables, Financial Services, FMCG, Healthcare, Information Technology, Metal,
Oil & Gas, Power, Realty, Telecommunication. All financials (banks, NBFCs,
insurers) go in Financial Services, and oil, gas and fuels in Oil & Gas. Index
names such as "Bankex" are rejected.

**The factor file's market series** is inspected rather than trusted. The
engine always uses benchmark minus cash as the market factor, so its
coefficient reads as active beta. A style series that looks like an index
return, rather than a long-short tilt, is converted to a tilt, and the change
is recorded.

**Bad input fails loudly.** The engine never silently coerces anything. A
wrong sector name, a percentage where a decimal belongs, a timezone on a date,
or weights that don't sum to 1 each stop the run with a message naming the
file, the column and the row.

---

## What the engine refuses to do

The validation gate checks every result before it is returned or written. If
any check fails, the run stops with a message naming the failure, **nothing is
written, and the last good file stays in place.** The gate refuses a result
when:

1. the four decision layers differ from gross active return by more than 1e-9;
2. gross active return plus the cost line differs from net active return by more than 1e-9;
3. the factor view's total differs from net active return by more than 1e-9;
4. portfolio or benchmark weights fail to sum to 1.0 within 1e-6 in any month;
5. any output figure is NaN or infinite;
6. the period has fewer than 24 months;
7. the policy's equity weight lies outside the policy's own band;
8. a sector appears on only one side (held but not in the benchmark, or the
   reverse) without being recorded as an orphan sector;
9. the macro layer contains anything other than exposure timing.

The reason is simple: an attribution whose parts do not add up to its total
cannot be explained to a client, so it is never shown.

**Data consistency.** The engine recomputes the portfolio return from holdings
x prices and compares it with the reported return. If they differ by more than
0.5% a month on average, the gap is flagged as a *material* limitation saying
the two inputs may not describe the same portfolio. It is **not** labelled as
fees.

---

## Known limitations (standing)

These apply to every run. Each run also generates its own list, with id,
severity (`info`, `caution`, `material`), text and the sections affected.

* **Short histories are noisy.** Five years of monthly data is 60
  observations. Most lines will have 95% intervals spanning zero, and the
  engine says so line by line. A clear exposure (such as a size tilt) can
  coexist with an uncertain contribution.
* **Few de-risking episodes.** Evidence about an overlay rests on a handful of
  events; the engine flags fewer than three.
* **Returns-based factor exposures are averages.** Exposures that drift over
  time are blurred. A 36-month rolling path is available for charting and is
  never used for totals.
* **The factor order matters** for individual factor lines, never for the total.
* **No market-timing term.** A Henriksson-Merton or Treynor-Mazuy term would
  split the market line into average beta and timing; it is not implemented.
* **Monthly exposure is a time-average** of the daily regime exposure, which is
  exact only if returns are spread evenly through the month.
* **Fees and trading costs are one line.** The data cannot separate
  management fees from trading costs or intra-month trading.
* **Long-only, single currency (INR).**
* **Confidence intervals** come from a block bootstrap (2,000 resamples,
  6-month blocks) on annualised arithmetic means. The headline lines are
  linked cumulative figures.

---

## Modules

| Module | Purpose |
|---|---|
| `schemas` | Input contracts: columns, units, the 13 sectors, the Policy; rejects bad input naming the row |
| `io` | Reads and writes CSV, Parquet and policy JSON, validating everything |
| `sample` | Generates the synthetic sample portfolio, benchmark, factors, cash and regime |
| `stats` | Block-bootstrap confidence intervals |
| `macro_layer` | The exposure-timing effect against the static policy weight, with its two-phase split and risk metrics |
| `brinson` | Allocation, selection and interaction (Brinson-Fachler or Hood-Beebower), scaling, and Carino/Menchero linking |
| `factors` | Returns-based factor model: series inspection, orthogonalisation, contributions, diagnostics, rolling exposures |
| `result` | The contract-level result the app reads (schema 1.1.0) |
| `outputs` | The published Result: attribution, gross-to-net bridge, reconciliation detail, checks, limitations, run info |
| `validate` | The gate that refuses to publish a broken result |
| `engine` | `run_attribution`: the single entry point |
| `cli` | `python -m attribution run ...` and the never-overwrite file writer |

Tests are in `tests/`; run them with `pytest`.
