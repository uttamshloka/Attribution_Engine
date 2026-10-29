"""Holdings-based sector attribution: allocation, selection and interaction.

Single-period effects (per sector i, per period t)
--------------------------------------------------
Brinson-Fachler (default)::

    allocation_i       = (w_p,i - w_b,i) * (r_b,i - r_b)
    selection_effect_i = w_b,i * (r_p,i - r_b,i)
    interaction_i      = (w_p,i - w_b,i) * (r_p,i - r_b,i)

``r_b`` is the TOTAL benchmark return for the period. Measuring each sector's
return against it means an overweight only scores when the sector beat the
benchmark, not merely when it went up.

Brinson-Hood-Beebower (``variant="hood_beebower"``) uses
``allocation_i = (w_p,i - w_b,i) * r_b,i``. Because active weights sum to
zero, the allocation TOTAL is the same; only the split across sectors
differs (every overweight looks good in a rising market). Both variants
give the same selection, interaction and total.

Across sectors the three effects sum to the EQUITY BOOK's active return,
``r_E - r_b``, where ``r_E`` is the sleeve's own return.

Terminology
-----------
"Selection" here is the MEASURED within-sector stock-picking effect, held in
columns named ``selection_effect``. It is not an unexplained residual: the
factor view calls its leftover "specific return". Do not reuse
``selection_effect`` for a residual anywhere in the package.

Joining the macro layer
-----------------------
With ``w_eq`` the portfolio's equity weight, ``w_pol`` the static policy
weight, ``r_c`` cash, and the holdings-based portfolio return
``r_P = w_eq * r_E + (1 - w_eq) * r_c``, the exact per-period identity is::

    r_P - r_policy = (w_eq - w_pol) * (r_b - r_c)        # macro layer
                   + w_eq * sum_i(allocation_i + selection_effect_i + interaction_i)

    r_policy = w_pol * r_b + (1 - w_pol) * r_c           # the policy benchmark

With a 100%-equity policy, ``r_policy`` is just ``r_b``. The Brinson effects
must therefore be SCALED by ``w_eq`` before being added to the macro
contribution; this module returns raw and scaled effects.

``r_P`` is the GROSS holdings-based return. Reported portfolio returns are
net of fees; the gap (fees, intra-period trading) is reported separately as
the residual and is not hidden inside any effect.

Conventions for missing pieces (all recorded, none silent)
----------------------------------------------------------
* Sector held but not in the benchmark (``w_b = 0``): ``r_b,i`` is set to
  ``r_p,i``, so the whole bet is allocation; selection and interaction are 0.
* Sector in the benchmark but not held (``w_p = 0``): ``r_p,i`` is set to
  ``r_b,i``, so selection and interaction are 0; allocation is the cost or
  benefit of being out.
* Zero weight in both: every effect is exactly 0.
* A held security with no price row for the period is EXCLUDED and listed in
  ``dropped``; the remaining weights in that period are rescaled to sum to 1
  (the dropped weight is reported, never filled with an imputed return).

Multi-period linking
--------------------
Arithmetic effects do not compound. Carino (default)::

    k_t = (ln(1+Rp_t) - ln(1+Rb_t)) / (Rp_t - Rb_t)   (1/(1+Rp_t) if equal)
    K   = same formula on the full-period returns
    adjusted_t = effect_t * k_t / K

Menchero is the alternative. Linking is applied to the SCALED effects
together with the macro contribution, so the linked effects sum to the
geometric active return ``prod(1+Rp) - prod(1+Rpolicy)``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from attribution.macro_layer import MacroResult, benchmark_total_return, compute_macro_contribution
from attribution.schemas import BENCHMARK, CASH_RETURN, HOLDINGS, PORTFOLIO_RETURN, PRICES, SECTORS, Policy
from attribution.stats import BootstrapResult, block_bootstrap

__all__ = [
    "EFFECTS",
    "Variant",
    "LinkMethod",
    "BrinsonError",
    "DroppedHolding",
    "OrphanSector",
    "SinglePeriodEffects",
    "LinkedEffects",
    "DecisionAttribution",
    "single_period_effects",
    "scale_by_exposure",
    "link_multi_period",
    "linking_factors",
    "by_sector_summary",
    "run_decision_attribution",
]

#: The three Brinson effects, by column name.
EFFECTS: tuple[str, ...] = ("allocation", "selection_effect", "interaction")
Variant = Literal["fachler", "hood_beebower"]
LinkMethod = Literal["carino", "menchero"]
_PERIODS_PER_YEAR = 12
_COMPLETE_TOL = 1e-10
_EQUAL_TOL = 1e-15


class BrinsonError(ValueError):
    """Raised when attribution inputs are inconsistent."""


@dataclass(frozen=True)
class DroppedHolding:
    """A holding excluded because its security had no price for the period."""

    date: date
    security_id: str
    sector: str
    weight: float
    reason: str


@dataclass(frozen=True)
class OrphanSector:
    """A sector held but not in the benchmark, or in the benchmark but not held, in one period."""

    period: str
    sector: str
    kind: Literal["held_not_in_benchmark", "benchmark_not_held"]
    portfolio_weight: float
    benchmark_weight: float
    convention: str


@dataclass(frozen=True)
class SinglePeriodEffects:
    """Per-sector, per-period effects for the equity book.

    ``table`` has one row per (period, sector) for all 15 sectors, with
    columns w_p, w_b, r_p, r_b, r_b_total and the effects. ``r_p`` / ``r_b``
    are after the orphan conventions; ``held`` / ``in_benchmark`` say which
    side actually had a position.
    """

    variant: str
    table: pd.DataFrame = field(repr=False)
    dropped: tuple[DroppedHolding, ...]
    orphans: tuple[OrphanSector, ...]

    def per_period(self) -> pd.DataFrame:
        """Effects summed over sectors per period, with the equity book's returns."""
        t = self.table
        out = t.groupby("period")[list(EFFECTS)].sum()
        out["r_equity_book"] = (t["w_p"] * t["r_p"]).groupby(t["period"]).sum()
        out["r_benchmark"] = t.groupby("period")["r_b_total"].first()
        out["active_equity_book"] = out["r_equity_book"] - out["r_benchmark"]
        out["total_effects"] = out[list(EFFECTS)].sum(axis=1)
        return out


# ---------------------------------------------------------------------------
# 1. Single-period effects
# ---------------------------------------------------------------------------


def single_period_effects(
    holdings: pd.DataFrame,
    benchmark: pd.DataFrame,
    prices: pd.DataFrame,
    variant: Variant = "fachler",
) -> SinglePeriodEffects:
    """Brinson effects per sector and period for the equity book."""
    if variant not in ("fachler", "hood_beebower"):
        raise BrinsonError(f"variant must be 'fachler' or 'hood_beebower', got {variant!r}")
    h = HOLDINGS.validate(holdings, source="holdings")
    b = BENCHMARK.validate(benchmark, source="benchmark")
    p = PRICES.validate(prices, source="prices")

    h_periods = set(h["date"].dt.to_period("M"))
    b_periods = set(b["date"].dt.to_period("M"))
    if h_periods != b_periods:
        raise BrinsonError(
            f"holdings and benchmark cover different months: only holdings {sorted(map(str, h_periods - b_periods))}, "
            f"only benchmark {sorted(map(str, b_periods - h_periods))}"
        )
    for name, frame in (("holdings", h), ("benchmark", b)):
        per = frame["date"].dt.to_period("M")
        if (frame.groupby(per)["date"].nunique() > 1).any():
            raise BrinsonError(f"{name} has more than one date in some month; supply one date per month")

    # Join prices; drop and record holdings with no price.
    merged = h.merge(p, on=["date", "security_id"], how="left", validate="one_to_one")
    missing = merged["total_return"].isna()
    dropped = tuple(
        DroppedHolding(r.date.date(), r.security_id, r.sector, float(r.weight), "no price row for this date")
        for r in merged.loc[missing].itertuples(index=False)
    )
    kept = merged.loc[~missing].copy()
    lost = sorted({str(d.date()) for d in h["date"]} - {str(d.date()) for d in kept["date"]})
    if lost:
        raise BrinsonError(f"every holding lacks a price on {lost}; cannot attribute those periods")
    kept["weight"] = kept["weight"] / kept.groupby("date")["weight"].transform("sum")
    kept["period"] = kept["date"].dt.to_period("M")
    kept["wr"] = kept["weight"] * kept["total_return"]
    port = kept.groupby(["period", "sector"])[["weight", "wr"]].sum()

    b = b.assign(period=b["date"].dt.to_period("M")).set_index(["period", "sector"])
    periods = sorted(h_periods)
    index = pd.MultiIndex.from_product([periods, list(SECTORS)], names=["period", "sector"])
    t = pd.DataFrame(index=index)
    t["w_p"] = port["weight"].reindex(index).fillna(0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        t["r_p"] = (port["wr"] / port["weight"]).reindex(index)
    t["w_b"] = b["weight"].reindex(index).fillna(0.0)
    t["r_b"] = b["total_return"].reindex(index)
    t = t.reset_index()
    t["held"] = t["w_p"] > 0
    t["in_benchmark"] = t["w_b"] > 0
    t["r_b_total"] = (t["w_b"] * t["r_b"].fillna(0.0)).groupby(t["period"]).transform("sum")

    orphans: list[OrphanSector] = []
    held_only = t["held"] & ~t["in_benchmark"]
    bench_only = ~t["held"] & t["in_benchmark"]
    neither = ~t["held"] & ~t["in_benchmark"]
    for r in t.loc[held_only].itertuples(index=False):
        orphans.append(OrphanSector(str(r.period), r.sector, "held_not_in_benchmark", float(r.w_p), 0.0,
                                    "r_b,i set to r_p,i: whole bet counted as allocation"))
    for r in t.loc[bench_only].itertuples(index=False):
        orphans.append(OrphanSector(str(r.period), r.sector, "benchmark_not_held", 0.0, float(r.w_b),
                                    "r_p,i set to r_b,i: selection and interaction are zero"))
    t.loc[held_only, "r_b"] = t.loc[held_only, "r_p"]
    t.loc[bench_only, "r_p"] = t.loc[bench_only, "r_b"]
    fill = t.loc[neither, "r_b"].fillna(t.loc[neither, "r_b_total"])
    t.loc[neither, "r_b"] = fill
    t.loc[neither, "r_p"] = fill

    active_w = t["w_p"] - t["w_b"]
    if variant == "fachler":
        t["allocation"] = active_w * (t["r_b"] - t["r_b_total"])
    else:
        t["allocation"] = active_w * t["r_b"]
    t["selection_effect"] = t["w_b"] * (t["r_p"] - t["r_b"])
    t["interaction"] = active_w * (t["r_p"] - t["r_b"])
    for col in EFFECTS:
        t.loc[neither, col] = 0.0
    t["total"] = t[list(EFFECTS)].sum(axis=1)

    if t[["r_p", "r_b", *EFFECTS]].isna().any().any():
        raise BrinsonError("internal error: unresolved missing return after conventions")
    return SinglePeriodEffects(variant=variant, table=t, dropped=dropped, orphans=tuple(orphans))


# ---------------------------------------------------------------------------
# 2. Scaling by equity exposure
# ---------------------------------------------------------------------------


def _to_period_series(s: pd.Series, name: str) -> pd.Series:
    idx = s.index
    if isinstance(idx, pd.PeriodIndex):
        out = s.copy()
    else:
        out = pd.Series(s.to_numpy(), index=pd.DatetimeIndex(idx).to_period("M"))
    out.name = name
    return out.astype(float)


def scale_by_exposure(effects: SinglePeriodEffects, equity_weights: pd.Series) -> pd.DataFrame:
    """Effects multiplied by each period's equity weight ``w_eq``.

    Returns the effects table with ``equity_weight`` and ``allocation_scaled``,
    ``selection_effect_scaled``, ``interaction_scaled`` added; the raw columns
    stay as they are.
    """
    w = _to_period_series(equity_weights, "equity_weight")
    t = effects.table.copy()
    missing = sorted({str(p) for p in t["period"]} - {str(p) for p in w.index})
    if missing:
        raise BrinsonError(f"equity_weights missing for period(s) {missing}")
    if ((w < 0) | (w > 1)).any():
        raise BrinsonError("equity weights must be within [0, 1]")
    t["equity_weight"] = t["period"].map(w)
    for col in EFFECTS:
        t[f"{col}_scaled"] = t[col] * t["equity_weight"]
    return t


# ---------------------------------------------------------------------------
# 3. Multi-period linking
# ---------------------------------------------------------------------------


def _log_ratio(rp: float, rb: float) -> float:
    if abs(rp - rb) < _EQUAL_TOL:
        return 1.0 / (1.0 + rp)
    return (math.log1p(rp) - math.log1p(rb)) / (rp - rb)


def linking_factors(portfolio_returns: pd.Series, benchmark_returns: pd.Series, method: LinkMethod = "carino") -> pd.Series:
    """Per-period multipliers that turn arithmetic effects into linked ones."""
    rp = portfolio_returns.to_numpy(dtype=float)
    rb = benchmark_returns.to_numpy(dtype=float)
    if method == "carino":
        RP, RB = float(np.prod(1 + rp) - 1), float(np.prod(1 + rb) - 1)
        K = _log_ratio(RP, RB)
        factors = np.array([_log_ratio(a, c) for a, c in zip(rp, rb, strict=True)]) / K
    elif method == "menchero":
        T = len(rp)
        RP, RB = float(np.prod(1 + rp) - 1), float(np.prod(1 + rb) - 1)
        A = RP - RB
        if abs(A) < _EQUAL_TOL:
            M = (1 + RP) ** (1 - 1 / T)
        else:
            M = (A / T) / ((1 + RP) ** (1 / T) - (1 + RB) ** (1 / T))
        a = rp - rb
        ss = float((a**2).sum())
        alpha = 0.0 if ss == 0.0 else (A - M * float(a.sum())) / ss
        factors = M + alpha * a
    else:
        raise BrinsonError(f"method must be 'carino' or 'menchero', got {method!r}")
    return pd.Series(factors, index=portfolio_returns.index, name=f"{method}_factor")


@dataclass(frozen=True)
class LinkedEffects:
    method: str
    factors: pd.Series = field(repr=False)
    adjusted: pd.DataFrame = field(repr=False)
    totals: Mapping[str, float]
    arithmetic_totals: Mapping[str, float]
    portfolio_cumulative: float
    benchmark_cumulative: float

    @property
    def geometric_active(self) -> float:
        return self.portfolio_cumulative - self.benchmark_cumulative

    @property
    def linked_sum(self) -> float:
        return float(sum(self.totals.values()))


def link_multi_period(
    effects: pd.DataFrame,
    method: LinkMethod = "carino",
    *,
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> LinkedEffects:
    """Link per-period effects so they sum to the geometric active return.

    ``effects`` is indexed by period with one column per effect; its columns
    must sum, in every period, to ``portfolio - benchmark`` for that period
    (to 1e-10), or linking is refused: linking an incomplete set would put
    the missing part into the others.
    """
    rp = _to_period_series(portfolio_returns, "portfolio")
    rb = _to_period_series(benchmark_returns, "benchmark")
    if not (effects.index.equals(rp.index) and rp.index.equals(rb.index)):
        raise BrinsonError("effects, portfolio_returns and benchmark_returns must cover the same periods in order")
    gap = (effects.sum(axis=1) - (rp - rb)).abs()
    if (gap > _COMPLETE_TOL).any():
        worst = gap.idxmax()
        raise BrinsonError(
            f"effects do not sum to the period's active return (worst {worst}: off by {gap.max():.3e}); "
            "include every component (e.g. the macro layer) before linking"
        )
    f = linking_factors(rp, rb, method)
    adjusted = effects.mul(f, axis=0)
    return LinkedEffects(
        method=method,
        factors=f,
        adjusted=adjusted,
        totals={c: float(adjusted[c].sum()) for c in effects.columns},
        arithmetic_totals={c: float(effects[c].sum()) for c in effects.columns},
        portfolio_cumulative=float((1 + rp).prod() - 1),
        benchmark_cumulative=float((1 + rb).prod() - 1),
    )


# ---------------------------------------------------------------------------
# 4. By-sector summary
# ---------------------------------------------------------------------------


def by_sector_summary(scaled: pd.DataFrame, factors: pd.Series) -> pd.DataFrame:
    """One row per sector plus TOTAL.

    w_p, w_b: average weights over all periods. r_p, r_b: compounded sector
    returns over the periods the side actually held the sector (blank if never).
    allocation, selection_effect, interaction: SCALED effects, linked with the
    same per-period factors as the totals, summed over periods.
    """
    t = scaled.copy()
    t["factor"] = t["period"].map(factors)
    rows = []
    for sector, g in t.groupby("sector", sort=False):
        rp = (1 + g.loc[g["held"], "r_p"]).prod() - 1 if g["held"].any() else np.nan
        rb = (1 + g.loc[g["in_benchmark"], "r_b"]).prod() - 1 if g["in_benchmark"].any() else np.nan
        rows.append({
            "sector": sector,
            "w_p": g["w_p"].mean(), "w_b": g["w_b"].mean(),
            "r_p": rp, "r_b": rb,
            **{c: float((g[f"{c}_scaled"] * g["factor"]).sum()) for c in EFFECTS},
        })
    out = pd.DataFrame(rows).set_index("sector")
    per = t.groupby("period")
    r_eq = (t["w_p"] * t["r_p"]).groupby(t["period"]).sum()
    r_bm = per["r_b_total"].first()
    total = {
        "w_p": out["w_p"].sum(), "w_b": out["w_b"].sum(),
        "r_p": float((1 + r_eq).prod() - 1), "r_b": float((1 + r_bm).prod() - 1),
        **{c: out[c].sum() for c in EFFECTS},
    }
    out.loc["TOTAL"] = total
    out["total"] = out[list(EFFECTS)].sum(axis=1)
    return out


# ---------------------------------------------------------------------------
# 5. Putting it together with the macro layer
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecisionAttribution:
    """Macro + allocation + selection + interaction, linked, for one portfolio."""

    variant: str
    method: str
    effects: SinglePeriodEffects
    scaled: pd.DataFrame = field(repr=False)
    macro: MacroResult = field(repr=False)
    per_period: pd.DataFrame = field(repr=False)
    linked: LinkedEffects = field(repr=False)
    by_sector: pd.DataFrame = field(repr=False)
    intervals: Mapping[str, BootstrapResult]
    residual_vs_reported: pd.Series | None = field(repr=False)

    @property
    def n_periods(self) -> int:
        return len(self.per_period)

    def annualised(self, cumulative: float) -> float:
        return (1 + cumulative) ** (_PERIODS_PER_YEAR / self.n_periods) - 1

    def report(self) -> str:
        L = self.linked

        def pct(x: float) -> str:
            return f"{x * 100:+.3f}%"

        ann_p, ann_b = self.annualised(L.portfolio_cumulative), self.annualised(L.benchmark_cumulative)
        lines = [
            "DECISION VIEW - holdings-based attribution",
            f"  Brinson variant {self.variant}; linking {L.method}; {self.n_periods} periods "
            f"({self.per_period.index[0]} to {self.per_period.index[-1]})",
            f"  portfolio (gross, holdings-based) {pct(L.portfolio_cumulative)} cumulative, {pct(ann_p)} a year",
            f"  policy benchmark                  {pct(L.benchmark_cumulative)} cumulative, {pct(ann_b)} a year",
            f"  geometric active return           {pct(L.geometric_active)} cumulative, {pct(ann_p - ann_b)} a year (difference of annualised)",
            "",
            "LINKED LAYERS (sum to the geometric active return)",
        ]
        for name in ("macro", *EFFECTS):
            ci = self.intervals.get(name)
            ci_txt = ""
            if ci is not None:
                ci_txt = f"   annualised 95% CI [{ci.lower * 100:+.2f}%, {ci.upper * 100:+.2f}%]" + (
                    "  ** spans zero **" if ci.spans_zero else "")
            lines.append(f"  {name:18s} {pct(L.totals[name])}{ci_txt}")
        lines += [
            f"  {'sum':18s} {pct(L.linked_sum)}   (geometric active {pct(L.geometric_active)}, "
            f"difference {L.linked_sum - L.geometric_active:.1e})",
            "",
        ]
        if self.residual_vs_reported is not None:
            res = self.residual_vs_reported
            lines.append(f"  reported (net) minus gross holdings-based return: {pct(float(res.sum()))} summed over periods "
                         f"(fees and trading; kept separate, not inside any effect)")
        lines.append(f"  dropped holdings (no price): {len(self.effects.dropped)}")
        held_only = sum(o.kind == "held_not_in_benchmark" for o in self.effects.orphans)
        bench_only = sum(o.kind == "benchmark_not_held" for o in self.effects.orphans)
        lines.append(f"  orphan sector-periods: {held_only} held but not in benchmark, {bench_only} in benchmark but not held")
        return "\n".join(lines)


def run_decision_attribution(
    holdings: pd.DataFrame,
    benchmark: pd.DataFrame,
    prices: pd.DataFrame,
    regime: pd.DataFrame,
    policy: Policy,
    cash_returns: pd.DataFrame,
    portfolio_returns: pd.DataFrame | None = None,
    *,
    variant: Variant = "fachler",
    method: LinkMethod = "carino",
    bootstrap: bool = True,
    seed: int = 20261028,
) -> DecisionAttribution:
    """Macro layer + Brinson effects, scaled, checked and linked."""
    effects = single_period_effects(holdings, benchmark, prices, variant)
    r_b_series = benchmark_total_return(benchmark)
    macro = compute_macro_contribution(regime, policy, r_b_series, cash_returns, bootstrap=bootstrap, seed=seed)
    w_eq = macro.per_period["equity_weight"]
    scaled = scale_by_exposure(effects, w_eq)

    pp = effects.per_period()
    cash = CASH_RETURN.validate(cash_returns, source="cash_returns")
    r_c = pd.Series(cash["total_return"].to_numpy(), index=cash["date"].dt.to_period("M"))
    w_pol = policy.neutral_equity_exposure
    frame = pd.DataFrame(index=pp.index)
    frame["equity_weight"] = w_eq.reindex(pp.index)
    frame["policy_weight"] = w_pol
    frame["r_cash"] = r_c.reindex(pp.index)
    frame["r_equity_book"] = pp["r_equity_book"]
    frame["r_benchmark"] = pp["r_benchmark"]
    frame["r_policy"] = w_pol * frame["r_benchmark"] + (1 - w_pol) * frame["r_cash"]
    frame["r_portfolio_gross"] = frame["equity_weight"] * frame["r_equity_book"] + (1 - frame["equity_weight"]) * frame["r_cash"]
    if frame.isna().any().any():
        raise BrinsonError("cash returns or equity weights do not cover every holdings period")
    frame["macro"] = macro.per_period["contribution"].reindex(pp.index)
    for col in EFFECTS:
        frame[col] = scaled.groupby("period")[f"{col}_scaled"].sum().reindex(pp.index)
    frame["identity_gap"] = (frame["r_portfolio_gross"] - frame["r_policy"]) - frame[["macro", *EFFECTS]].sum(axis=1)

    linked = link_multi_period(
        frame[["macro", *EFFECTS]], method,
        portfolio_returns=frame["r_portfolio_gross"], benchmark_returns=frame["r_policy"],
    )
    sectors = by_sector_summary(scaled, linked.factors)

    intervals: dict[str, BootstrapResult] = {}
    if bootstrap and len(frame) >= 2:
        if macro.ci is not None:
            intervals["macro"] = macro.ci
        for col in EFFECTS:
            intervals[col] = block_bootstrap(frame[col].to_numpy(), lambda a: float(a.mean()) * _PERIODS_PER_YEAR,
                                             frequency="monthly", seed=seed)

    residual = None
    if portfolio_returns is not None:
        rep = PORTFOLIO_RETURN.validate(portfolio_returns, source="portfolio_returns")
        rep_s = pd.Series(rep["total_return"].to_numpy(), index=rep["date"].dt.to_period("M")).reindex(pp.index)
        if rep_s.isna().any():
            raise BrinsonError("portfolio_returns do not cover every holdings period")
        residual = (rep_s - frame["r_portfolio_gross"]).rename("reported_minus_gross")
        frame["r_portfolio_reported"] = rep_s

    return DecisionAttribution(
        variant=variant, method=method, effects=effects, scaled=scaled, macro=macro, per_period=frame,
        linked=linked, by_sector=sectors, intervals=intervals, residual_vs_reported=residual,
    )
