"""Returns-based factor attribution (the factor view).

Model
-----
Dependent variable: monthly ACTIVE return ``y_t = r_portfolio,t - r_benchmark,t``
(portfolio net of fees, as reported). Regressors, in this pre-committed order::

    market, size, value, momentum, quality, low_volatility

The market regressor is the benchmark's EXCESS return ``r_benchmark - r_cash``,
so its coefficient reads as ACTIVE beta (beta_portfolio - 1): the
exposure-timing effect, comparable with the decision view's macro layer.

Headline decomposition: one full-period OLS with an intercept. Because OLS
residuals have mean zero, the identity is exact::

    mean(y) = alpha + sum_k beta_k * mean(f_k)
    contribution_k  = beta_k * mean(f_k) * 12
    specific_return = alpha * 12
    sum_k contribution_k + specific_return = mean(y) * 12   (annualised active)

"Specific return" is the factor view's leftover. It is NOT the decision
view's ``selection_effect``, which is a measured stock-picking effect.

Collinearity
------------
Factor series from overlapping index families are often highly correlated,
and a naive regression then gives unstable, sign-flipping exposures. The
factors are orthogonalised by sequential Gram-Schmidt in the order above:
each factor keeps only the part not already explained by the factors before
it. The order changes the individual answers, so it is recorded, and each
factor's contribution is INCREMENTAL to those before it.

The projection uses plain (uncentred) inner products, so the transformation
is a pure triangular change of basis with no constant. As a result the
fitted values, the intercept (specific return) and the total of the factor
contributions are all unchanged by orthogonalisation or by the order; only
the split among factors moves.

Diagnostics: condition number of X'X computed on the standardised raw factors
(i.e. of their correlation matrix, so it does not depend on units), VIF per
factor, R-squared, adjusted R-squared and Durbin-Watson. A condition number
above 30 adds an entry to the limitations list.

Rolling exposures
-----------------
:func:`rolling_exposures` fits the same model on a moving 36-month window to
chart how exposures drift. It is SEPARATE from the headline decomposition and
is never used for any total.

Seam for holdings-based factors
-------------------------------
:class:`FactorModelResult` records ``method="returns_based"``. A future
holdings-based estimator should produce the same result type (exposures,
contributions, specific return, diagnostics) so the app and the
reconciliation code do not change.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from attribution.brinson import LinkedEffects, link_multi_period
from attribution.macro_layer import benchmark_total_return
from attribution.result import FactorContribution, FactorDiagnostics, FactorView, SpecificReturn
from attribution.schemas import CASH_RETURN, FACTOR_RETURNS, FACTORS, PORTFOLIO_RETURN
from attribution.stats import BootstrapResult, block_bootstrap

__all__ = [
    "FACTOR_ORDER",
    "CONDITION_LIMIT",
    "IDENTITY_TOL",
    "MARKET_VS_MACRO_NOTE",
    "FactorError",
    "SeriesInspection",
    "FactorInspection",
    "FactorMatrix",
    "Orthogonalised",
    "FactorRow",
    "FactorModelResult",
    "inspect_factor_series",
    "build_factor_matrix",
    "orthogonalise",
    "active_returns",
    "fit_factor_model",
    "rolling_exposures",
    "link_factor_contributions",
    "collinearity_diagnostics",
]

#: Pre-committed orthogonalisation order.
FACTOR_ORDER: tuple[str, ...] = FACTORS
CONDITION_LIMIT = 30.0
IDENTITY_TOL = 1e-10
_PERIODS_PER_YEAR = 12

MARKET_VS_MACRO_NOTE = (
    "The factor view's market-exposure line and the decision view's macro layer measure the same economic "
    "decision - how much market risk to carry - by different routes: a monthly regression of active returns "
    "versus period-by-period holdings and exposure weights. They will be close but not identical. The TOTALS "
    "of the two views reconcile; individual lines across the two views do not have to."
)

Classification = Literal[
    "benchmark_itself", "excess_market_return", "index_total_return", "tilt_return", "ambiguous"
]
MarketSource = Literal["benchmark_excess", "supplied"]


class FactorError(ValueError):
    """Raised when factor inputs are inconsistent or degenerate."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _monthly(frame: pd.DataFrame, schema_source: str) -> pd.Series:
    clean = CASH_RETURN.validate(frame, source=schema_source)
    s = pd.Series(clean["total_return"].to_numpy(), index=clean["date"].dt.to_period("M"), name=schema_source)
    if s.index.duplicated().any():
        raise FactorError(f"{schema_source}: more than one row in some month")
    return s.sort_index()


def _benchmark_monthly(benchmark: pd.DataFrame) -> pd.Series:
    rb = benchmark_total_return(benchmark)
    return pd.Series(rb.to_numpy(), index=pd.DatetimeIndex(rb.index).to_period("M"), name="benchmark").sort_index()


def _factor_wide(factor_returns: pd.DataFrame) -> pd.DataFrame:
    clean = FACTOR_RETURNS.validate(factor_returns, source="factor_returns")
    wide = clean.assign(period=clean["date"].dt.to_period("M")).pivot(index="period", columns="factor", values="total_return")
    missing = [f for f in FACTORS if f not in wide.columns]
    if missing:
        raise FactorError(f"factor_returns is missing factor(s) {missing}")
    if wide.isna().any().any():
        gaps = [f"{c} {p}" for c in wide.columns for p in wide.index[wide[c].isna()]]
        raise FactorError(f"factor_returns has gaps: {gaps[:10]}")
    return wide[list(FACTORS)].sort_index()


def _same_months(named: Mapping[str, pd.Index]) -> pd.PeriodIndex:
    items = list(named.items())
    base_name, base = items[0]
    for name, idx in items[1:]:
        if not idx.equals(base):
            only_a = sorted(map(str, base.difference(idx)))
            only_b = sorted(map(str, idx.difference(base)))
            raise FactorError(f"{base_name} and {name} cover different months: only {base_name} {only_a}, only {name} {only_b}")
    return pd.PeriodIndex(base, freq="M")


# ---------------------------------------------------------------------------
# 1. Inspect the supplied series
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SeriesInspection:
    factor: str
    mean_annual: float
    vol_annual: float
    corr_benchmark: float
    slope_on_benchmark: float
    max_abs_diff_vs_benchmark: float
    classification: str
    handling: str


@dataclass(frozen=True)
class FactorInspection:
    rows: tuple[SeriesInspection, ...]
    benchmark_mean_annual: float
    cash_mean_annual: float
    n_months: int
    market_source: str

    def row(self, factor: str) -> SeriesInspection:
        for r in self.rows:
            if r.factor == factor:
                return r
        raise KeyError(factor)

    def report(self) -> str:
        lines = [
            "FACTOR SERIES INSPECTION",
            f"  {self.n_months} months; benchmark mean {self.benchmark_mean_annual * 100:+.2f}% a year, "
            f"cash {self.cash_mean_annual * 100:+.2f}%, benchmark minus cash "
            f"{(self.benchmark_mean_annual - self.cash_mean_annual) * 100:+.2f}%",
            f"  {'factor':15s} {'mean/yr':>8s} {'vol/yr':>7s} {'corr bm':>8s} {'slope':>6s} {'max|f-bm|':>9s}  classification",
        ]
        for r in self.rows:
            lines.append(
                f"  {r.factor:15s} {r.mean_annual * 100:+7.2f}% {r.vol_annual * 100:6.2f}% {r.corr_benchmark:+8.3f} "
                f"{r.slope_on_benchmark:+6.2f} {r.max_abs_diff_vs_benchmark * 100:8.2f}%  {r.classification}"
            )
        lines.append("  handling:")
        for r in self.rows:
            lines.append(f"    {r.factor:15s} {r.handling}")
        return "\n".join(lines)


def _classify(f: np.ndarray, rb: np.ndarray, rc: np.ndarray) -> tuple[Classification, float, float, float]:
    corr = float(np.corrcoef(f, rb)[0, 1]) if np.std(f) > 0 and np.std(rb) > 0 else 0.0
    slope = float(np.polyfit(rb, f, 1)[0]) if np.std(rb) > 0 else 0.0
    maxdiff = float(np.max(np.abs(f - rb)))
    if maxdiff < 1e-10:
        return "benchmark_itself", corr, slope, maxdiff
    if corr >= 0.9 and 0.75 <= slope <= 1.25:
        excess_gap = abs(f.mean() - (rb.mean() - rc.mean()))
        total_gap = abs(f.mean() - rb.mean())
        return ("excess_market_return" if excess_gap < total_gap else "index_total_return"), corr, slope, maxdiff
    if abs(corr) < 0.7:
        return "tilt_return", corr, slope, maxdiff
    return "ambiguous", corr, slope, maxdiff


def _handling(factor: str, cls: str, market_source: str) -> str:
    if factor == "market":
        if market_source == "benchmark_excess":
            return (f"supplied series looks like '{cls}'; NOT used. Market regressor = benchmark minus cash, "
                    "so its coefficient is active beta (beta_portfolio - 1)")
        if cls in ("index_total_return", "benchmark_itself"):
            return "supplied series is a total return: cash subtracted to make it an excess return"
        if cls == "excess_market_return":
            return "supplied series is already an excess return: used as is"
        raise FactorError(f"market series looks like '{cls}', not a market return; use market_source='benchmark_excess'")
    if cls == "tilt_return":
        return "long-short tilt return: used as is"
    if cls in ("index_total_return", "benchmark_itself", "excess_market_return"):
        return "looks like an index return, not a tilt: converted to a tilt by subtracting the benchmark return"
    return "AMBIGUOUS (neither clearly a tilt nor an index): used as is - see limitations"


def inspect_factor_series(
    factor_returns: pd.DataFrame,
    benchmark: pd.DataFrame,
    cash_returns: pd.DataFrame,
    *,
    market_source: MarketSource = "benchmark_excess",
    verbose: bool = True,
) -> FactorInspection:
    """Classify each supplied factor series as an index return or a tilt return.

    Prints the report when ``verbose`` (default) and returns it as an object.
    """
    if market_source not in ("benchmark_excess", "supplied"):
        raise FactorError(f"market_source must be 'benchmark_excess' or 'supplied', got {market_source!r}")
    wide = _factor_wide(factor_returns)
    rb = _benchmark_monthly(benchmark)
    rc = _monthly(cash_returns, "cash_returns")
    _same_months({"factor_returns": wide.index, "benchmark": rb.index, "cash_returns": rc.index})
    rows = []
    for factor in FACTORS:
        x = wide[factor].to_numpy()
        cls, corr, slope, maxdiff = _classify(x, rb.to_numpy(), rc.to_numpy())
        rows.append(SeriesInspection(
            factor=factor,
            mean_annual=float(x.mean() * _PERIODS_PER_YEAR),
            vol_annual=float(x.std(ddof=1) * np.sqrt(_PERIODS_PER_YEAR)),
            corr_benchmark=corr, slope_on_benchmark=slope, max_abs_diff_vs_benchmark=maxdiff,
            classification=cls, handling=_handling(factor, cls, market_source),
        ))
    out = FactorInspection(
        rows=tuple(rows),
        benchmark_mean_annual=float(rb.mean() * _PERIODS_PER_YEAR),
        cash_mean_annual=float(rc.mean() * _PERIODS_PER_YEAR),
        n_months=len(rb),
        market_source=market_source,
    )
    if verbose:
        print(out.report())
    return out


# ---------------------------------------------------------------------------
# 2. Regression matrix
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorMatrix:
    """Regressors in FACTOR_ORDER, indexed by month, with a record of every transformation."""

    data: pd.DataFrame = field(repr=False)
    transformations: Mapping[str, str]
    warnings: tuple[str, ...]
    inspection: FactorInspection


def build_factor_matrix(
    factor_returns: pd.DataFrame,
    benchmark: pd.DataFrame,
    cash_returns: pd.DataFrame,
    *,
    market_source: MarketSource = "benchmark_excess",
) -> FactorMatrix:
    """The regression matrix: market as an excess return, the rest as tilts."""
    inspection = inspect_factor_series(factor_returns, benchmark, cash_returns, market_source=market_source, verbose=False)
    wide = _factor_wide(factor_returns)
    rb = _benchmark_monthly(benchmark)
    rc = _monthly(cash_returns, "cash_returns")
    months = _same_months({"factor_returns": wide.index, "benchmark": rb.index, "cash_returns": rc.index})
    data = pd.DataFrame(index=months)
    warnings: list[str] = []
    for factor in FACTORS:
        r = inspection.row(factor)
        x = wide[factor]
        if factor == "market":
            if market_source == "benchmark_excess":
                data[factor] = rb - rc
            elif r.classification in ("index_total_return", "benchmark_itself"):
                data[factor] = x - rc
            else:
                data[factor] = x
        elif r.classification in ("index_total_return", "benchmark_itself", "excess_market_return"):
            data[factor] = x - rb
        else:
            data[factor] = x
            if r.classification == "ambiguous":
                warnings.append(f"factor '{factor}' could not be classified as a tilt or an index return "
                                f"(correlation with benchmark {r.corr_benchmark:+.2f}); used as supplied")
    data.index.name = "period"
    return FactorMatrix(
        data=data, transformations={r.factor: r.handling for r in inspection.rows},
        warnings=tuple(warnings), inspection=inspection,
    )


# ---------------------------------------------------------------------------
# 3. Orthogonalisation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Orthogonalised:
    """Gram-Schmidt output.

    ``data`` holds the orthogonalised factors (each orthogonal, in the plain
    inner-product sense, to every factor before it). ``projection`` holds the
    coefficients removed: ``f*_k = f_k - sum_j projection[k, j] * f*_j``.
    """

    data: pd.DataFrame = field(repr=False)
    order: tuple[str, ...]
    projection: pd.DataFrame = field(repr=False)

    @property
    def note(self) -> str:
        return ("factors orthogonalised by sequential Gram-Schmidt in the order " + " -> ".join(self.order)
                + "; each factor's contribution is incremental to those before it")


def _gram_schmidt(X: np.ndarray, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
    """Sequential (uncentred) Gram-Schmidt on the columns of X, in column order."""
    n, k = X.shape
    Q = np.empty_like(X)
    P = np.zeros((k, k))
    for i in range(k):
        v = X[:, i].copy()
        for j in range(i):
            P[i, j] = float(X[:, i] @ Q[:, j]) / float(Q[:, j] @ Q[:, j])
            v -= P[i, j] * Q[:, j]
        if float(v @ v) <= 1e-14 * max(1.0, float(X[:, i] @ X[:, i])):
            raise FactorError(f"factor '{names[i]}' is a linear combination of the factors before it; cannot orthogonalise")
        Q[:, i] = v
    return Q, P


def orthogonalise(factors: pd.DataFrame, order: Sequence[str] = FACTOR_ORDER) -> Orthogonalised:
    order = tuple(order)
    if sorted(order) != sorted(factors.columns):
        raise FactorError(f"order {order} must list each factor column exactly once: {list(factors.columns)}")
    Q, P = _gram_schmidt(factors[list(order)].to_numpy(dtype=float), order)
    return Orthogonalised(
        data=pd.DataFrame(Q, index=factors.index, columns=list(order)),
        order=order,
        projection=pd.DataFrame(P, index=list(order), columns=list(order)),
    )


# ---------------------------------------------------------------------------
# 4. Diagnostics
# ---------------------------------------------------------------------------


def collinearity_diagnostics(factors: pd.DataFrame) -> tuple[float, dict[str, float]]:
    """(condition number of X'X on standardised factors, VIF per factor)."""
    X = factors.to_numpy(dtype=float)
    Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=0)
    corr = (Z.T @ Z) / len(Z)
    eig = np.linalg.eigvalsh(corr)
    cond = float(eig.max() / eig.min()) if eig.min() > 0 else float("inf")
    vif: dict[str, float] = {}
    for i, name in enumerate(factors.columns):
        others = np.delete(X, i, axis=1)
        A = np.column_stack([np.ones(len(X)), others])
        coef, *_ = np.linalg.lstsq(A, X[:, i], rcond=None)
        resid = X[:, i] - A @ coef
        ss_tot = float(((X[:, i] - X[:, i].mean()) ** 2).sum())
        r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 1.0
        vif[name] = float("inf") if r2 >= 1.0 else 1.0 / (1.0 - r2)
    return cond, vif


def _ols(y: np.ndarray, X: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Intercept, slopes, residuals. Exact zero when y is identically zero."""
    if not np.any(y):
        return 0.0, np.zeros(X.shape[1]), np.zeros_like(y)
    A = np.column_stack([np.ones(len(y)), X])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(coef[0]), coef[1:], y - A @ coef


# ---------------------------------------------------------------------------
# 5. The model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FactorRow:
    factor: str
    exposure: float          # coefficient on the (orthogonalised) factor
    exposure_raw: float      # coefficient in the plain, non-orthogonalised regression
    t_stat: float
    factor_mean_annual: float
    contribution: float      # exposure * mean(factor) * 12
    contribution_raw: float
    ci: BootstrapResult | None            # on the annualised contribution
    exposure_ci: BootstrapResult | None   # on the exposure itself


@dataclass(frozen=True)
class FactorModelResult:
    method: str
    order: tuple[str, ...]
    orthogonalised: bool
    rows: tuple[FactorRow, ...]
    alpha_monthly: float
    specific_return: float
    specific_t_stat: float
    specific_ci: BootstrapResult | None
    active_annualised: float
    n_obs: int
    r_squared: float
    adj_r_squared: float
    durbin_watson: float
    condition_number_raw: float
    condition_number_used: float
    vif_raw: Mapping[str, float]
    vif_used: Mapping[str, float]
    tracking_error: float
    residual_volatility: float
    per_period: pd.DataFrame = field(repr=False)
    transformations: Mapping[str, str]
    limitations: tuple[str, ...]
    note: str = MARKET_VS_MACRO_NOTE

    @property
    def total(self) -> float:
        """Sum of factor contributions plus specific return (equals annualised active)."""
        return float(sum(r.contribution for r in self.rows) + self.specific_return)

    @property
    def identity_gap(self) -> float:
        return self.total - self.active_annualised

    def row(self, factor: str) -> FactorRow:
        for r in self.rows:
            if r.factor == factor:
                return r
        raise KeyError(factor)

    def to_factor_view(self) -> FactorView:
        """The ``factor_view`` block of AttributionResult (factors in contract order)."""
        by_name = {r.factor: r for r in self.rows}
        info = (self.specific_return / self.residual_volatility) if self.residual_volatility > 0 else None
        return FactorView(
            factors=tuple(
                FactorContribution(f, exposure=by_name[f].exposure, factor_return=by_name[f].factor_mean_annual,
                                   contribution=by_name[f].contribution, t_stat=by_name[f].t_stat)
                for f in FACTORS
            ),
            specific_return=SpecificReturn(value=self.specific_return, t_stat=self.specific_t_stat, information_ratio=info),
            diagnostics=FactorDiagnostics(
                method=f"{self.method}; OLS with intercept; Gram-Schmidt order {' -> '.join(self.order)}"
                if self.orthogonalised else f"{self.method}; OLS with intercept; not orthogonalised",
                n_obs=self.n_obs,
                r_squared=min(1.0, max(0.0, self.r_squared)),
                tracking_error=self.tracking_error,
                residual_volatility=self.residual_volatility,
                warnings=self.limitations,
            ),
        )

    def report(self) -> str:
        lines = [
            "FACTOR VIEW - returns-based",
            f"  dependent variable: monthly active return (portfolio net - benchmark), {self.n_obs} months",
            f"  method: {self.method}; OLS with intercept"
            + (f"; Gram-Schmidt order {' -> '.join(self.order)} (each contribution incremental to those before it)"
               if self.orthogonalised else "; NOT orthogonalised"),
            "",
            "  Exposure CI asks 'is the tilt there?'; contribution CI asks 'did it pay, with confidence?'.",
            "  A clear exposure with a wide contribution CI means the tilt is real but the factor's own",
            "  return over the window is too noisy to prove it paid.",
            "",
            f"  {'factor':15s} {'exposure':>9s} {'t':>6s} {'exposure 95% CI':>22s} {'raw expo':>9s} {'f mean/yr':>9s} "
            f"{'contrib/yr':>10s}   contribution 95% CI",
        ]

        def flag(c: BootstrapResult | None) -> str:
            return "*" if c is not None and c.spans_zero else " "

        for r in self.rows:
            eci = f"[{r.exposure_ci.lower:+.3f}, {r.exposure_ci.upper:+.3f}]{flag(r.exposure_ci)}" if r.exposure_ci else ""
            ci = f"[{r.ci.lower * 100:+.2f}%, {r.ci.upper * 100:+.2f}%]{flag(r.ci)}" if r.ci else ""
            lines.append(f"  {r.factor:15s} {r.exposure:+9.3f} {r.t_stat:+6.2f} {eci:>22s} {r.exposure_raw:+9.3f} "
                         f"{r.factor_mean_annual * 100:+8.2f}% {r.contribution * 100:+9.3f}%   {ci}")
        sci = ""
        if self.specific_ci is not None:
            sci = (f"[{self.specific_ci.lower * 100:+.2f}%, {self.specific_ci.upper * 100:+.2f}%]"
                   + ("*" if self.specific_ci.spans_zero else " "))
        lines += [
            f"  {'specific return':15s} {'':9s} {self.specific_t_stat:+6.2f} {'':22s} {'':9s} {'':9s} "
            f"{self.specific_return * 100:+9.3f}%   {sci}",
            f"  {'TOTAL':15s} {'':9s} {'':6s} {'':22s} {'':9s} {'':9s} {self.total * 100:+9.3f}%",
            "  * = interval spans zero (not statistically distinguishable from zero)",
            "",
            f"  RECONCILIATION: factors + specific = {self.total * 100:+.6f}% a year; annualised active "
            f"(mean x 12) = {self.active_annualised * 100:+.6f}%; gap {self.identity_gap:.1e} (must be below 1e-10)",
            "",
            "DIAGNOSTICS",
            f"  R-squared {self.r_squared:.3f}   adjusted {self.adj_r_squared:.3f}   Durbin-Watson {self.durbin_watson:.2f}",
            f"  condition number of X'X (standardised): raw factors {self.condition_number_raw:.2f}, "
            f"as used {self.condition_number_used:.2f} (limit {CONDITION_LIMIT:g})",
            "  VIF raw:     " + ", ".join(f"{k} {v:.2f}" for k, v in self.vif_raw.items()),
            "  VIF as used: " + ", ".join(f"{k} {v:.2f}" for k, v in self.vif_used.items()),
            f"  tracking error {self.tracking_error * 100:.2f}% a year; residual volatility {self.residual_volatility * 100:.2f}% a year",
            "",
            "TRANSFORMATIONS",
            *[f"  {k:15s} {v}" for k, v in self.transformations.items()],
            "",
            "NOTE",
            f"  {self.note}",
        ]
        if self.limitations:
            lines += ["", "LIMITATIONS", *[f"  - {x}" for x in self.limitations]]
        return "\n".join(lines)


def active_returns(portfolio_returns: pd.DataFrame, benchmark: pd.DataFrame) -> pd.Series:
    """Monthly active return: reported portfolio return (net) minus benchmark return."""
    rep = PORTFOLIO_RETURN.validate(portfolio_returns, source="portfolio_returns")
    rp = pd.Series(rep["total_return"].to_numpy(), index=rep["date"].dt.to_period("M")).sort_index()
    rb = _benchmark_monthly(benchmark)
    _same_months({"portfolio_returns": rp.index, "benchmark": rb.index})
    return (rp - rb).rename("active")


def _fit_core(y: np.ndarray, F: np.ndarray, order_idx: list[int], orth: bool) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha, betas in order, residuals, regressor matrix used)."""
    X = F[:, order_idx]
    if orth:
        X, _ = _gram_schmidt(X, [str(i) for i in order_idx])
    a, b, e = _ols(y, X)
    return a, b, e, X


def fit_factor_model(
    active: pd.Series,
    factors: FactorMatrix | pd.DataFrame,
    *,
    order: Sequence[str] = FACTOR_ORDER,
    orthogonalise_factors: bool = True,
    bootstrap: bool = True,
    seed: int = 20261028,
) -> FactorModelResult:
    """Full-period OLS decomposition of annualised active return into factors + specific."""
    if isinstance(factors, FactorMatrix):
        F_df, transformations, warnings = factors.data, dict(factors.transformations), list(factors.warnings)
    else:
        F_df, transformations, warnings = factors, {c: "supplied matrix used as is" for c in factors.columns}, []
    order = tuple(order)
    if sorted(order) != sorted(F_df.columns):
        raise FactorError(f"order {order} must list each factor exactly once: {list(F_df.columns)}")
    y_s = active if isinstance(active.index, pd.PeriodIndex) else pd.Series(
        active.to_numpy(), index=pd.DatetimeIndex(active.index).to_period("M"))
    months = _same_months({"active": y_s.index, "factors": F_df.index})
    y = y_s.to_numpy(dtype=float)
    F_used_df = F_df[list(order)]
    n, k = F_used_df.shape
    if n <= k + 1:
        raise FactorError(f"need more than {k + 1} months to fit {k} factors, got {n}")
    F = F_used_df.to_numpy(dtype=float)
    idx = list(range(k))

    alpha, beta, resid, X = _fit_core(y, F, idx, orthogonalise_factors)
    alpha_raw, beta_raw, _, _ = _fit_core(y, F, idx, False)
    means = X.mean(axis=0)
    contrib = beta * means * _PERIODS_PER_YEAR
    contrib_raw = beta_raw * F.mean(axis=0) * _PERIODS_PER_YEAR
    specific = alpha * _PERIODS_PER_YEAR
    active_ann = float(y.mean() * _PERIODS_PER_YEAR)
    gap = float(contrib.sum() + specific - active_ann)
    if abs(gap) > IDENTITY_TOL:
        raise FactorError(f"identity failed: factors + specific - active = {gap:.3e}")

    dof = n - k - 1
    sse = float(resid @ resid)
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = 0.0 if sst == 0 else 1.0 - sse / sst
    adj = 0.0 if sst == 0 else 1.0 - (1.0 - r2) * (n - 1) / dof
    dw = float((np.diff(resid) ** 2).sum() / sse) if sse > 0 else 2.0
    A = np.column_stack([np.ones(n), X])
    if sse > 0:
        cov = (sse / dof) * np.linalg.inv(A.T @ A)
        se = np.sqrt(np.diag(cov))
        t_alpha, t_beta = alpha / se[0], beta / se[1:]
    else:
        t_alpha, t_beta = 0.0, np.zeros(k)

    cond_raw, vif_raw = collinearity_diagnostics(F_used_df)
    cond_used, vif_used = collinearity_diagnostics(pd.DataFrame(X, columns=list(order)))
    limitations = list(warnings)
    if orthogonalise_factors:
        limitations.append("Factor contributions are incremental in the order " + " -> ".join(order)
                           + "; a different order would split the same total differently.")
    if cond_raw > CONDITION_LIMIT:
        limitations.append(f"Condition number of X'X for the raw factors is {cond_raw:.1f} (above {CONDITION_LIMIT:g}): "
                           "the factors are strongly collinear and raw exposures are unstable.")
    limitations.append("Returns-based exposures are averages over the whole window; exposures that change over "
                       "time are blurred (see rolling_exposures for the path).")

    cis: list[BootstrapResult | None] = [None] * k
    ecis: list[BootstrapResult | None] = [None] * k
    sci: BootstrapResult | None = None
    if bootstrap and np.any(y):
        positions = np.arange(n, dtype=float)

        def stat_for(j: int, what: str):
            def stat(ix: np.ndarray) -> float:
                ii = ix.astype(int)
                a_, b_, _, X_ = _fit_core(y[ii], F[ii], idx, orthogonalise_factors)
                if what == "specific":
                    return a_ * _PERIODS_PER_YEAR
                if what == "exposure":
                    return float(b_[j])
                return float(b_[j] * X_[:, j].mean() * _PERIODS_PER_YEAR)
            return stat

        for j in range(k):
            cis[j] = block_bootstrap(positions, stat_for(j, "contribution"), frequency="monthly", seed=seed)
            ecis[j] = block_bootstrap(positions, stat_for(j, "exposure"), frequency="monthly", seed=seed)
        sci = block_bootstrap(positions, stat_for(-1, "specific"), frequency="monthly", seed=seed)
        # The bootstrap point must equal the headline (it is computed on the original sample).
        for j in range(k):
            assert cis[j] is not None and abs(cis[j].point - contrib[j]) < 1e-12

    per = pd.DataFrame(X * beta, index=months, columns=list(order))
    per["specific"] = alpha + resid
    per["active"] = y

    rows = tuple(
        FactorRow(order[j], float(beta[j]), float(beta_raw[j]), float(t_beta[j]), float(means[j] * _PERIODS_PER_YEAR),
                  float(contrib[j]), float(contrib_raw[j]), cis[j], ecis[j])
        for j in range(k)
    )
    return FactorModelResult(
        method="returns_based",
        order=order,
        orthogonalised=orthogonalise_factors,
        rows=rows,
        alpha_monthly=alpha,
        specific_return=specific,
        specific_t_stat=float(t_alpha),
        specific_ci=sci,
        active_annualised=active_ann,
        n_obs=n,
        r_squared=r2,
        adj_r_squared=adj,
        durbin_watson=dw,
        condition_number_raw=cond_raw,
        condition_number_used=cond_used,
        vif_raw=vif_raw,
        vif_used=vif_used,
        tracking_error=float(y.std(ddof=1) * np.sqrt(_PERIODS_PER_YEAR)),
        residual_volatility=float(np.sqrt(sse / dof) * np.sqrt(_PERIODS_PER_YEAR)) if dof > 0 else 0.0,
        per_period=per,
        transformations=transformations,
        limitations=tuple(limitations),
    )


# ---------------------------------------------------------------------------
# 6. Rolling exposures (charting only)
# ---------------------------------------------------------------------------


def rolling_exposures(
    active: pd.Series,
    factors: FactorMatrix | pd.DataFrame,
    window: int = 36,
    *,
    order: Sequence[str] = FACTOR_ORDER,
    orthogonalise_factors: bool = True,
) -> pd.DataFrame:
    """Exposures from a moving window, for charting how they drift.

    One row per window end month: the exposures and the monthly intercept.
    NOT used for, and must never be used for, the headline totals.
    """
    F_df = factors.data if isinstance(factors, FactorMatrix) else factors
    order = tuple(order)
    y_s = active if isinstance(active.index, pd.PeriodIndex) else pd.Series(
        active.to_numpy(), index=pd.DatetimeIndex(active.index).to_period("M"))
    months = _same_months({"active": y_s.index, "factors": F_df.index})
    k = len(order)
    if window <= k + 1:
        raise FactorError(f"window must exceed {k + 1} months for {k} factors")
    if window > len(months):
        raise FactorError(f"window {window} is longer than the {len(months)} months of data")
    y = y_s.to_numpy(dtype=float)
    F = F_df[list(order)].to_numpy(dtype=float)
    rows = []
    for end in range(window, len(months) + 1):
        a, b, _, _ = _fit_core(y[end - window:end], F[end - window:end], list(range(k)), orthogonalise_factors)
        rows.append([a, *b])
    out = pd.DataFrame(rows, index=months[window - 1:], columns=["alpha_monthly", *order])
    out.index.name = "window_end"
    return out


# ---------------------------------------------------------------------------
# 7. Linking, for reconciling with the decision view
# ---------------------------------------------------------------------------


def link_factor_contributions(
    model: FactorModelResult,
    portfolio_returns: pd.DataFrame,
    benchmark: pd.DataFrame,
    method: Literal["carino", "menchero"] = "carino",
) -> LinkedEffects:
    """Link per-period factor contributions (+ specific incl. residual) so they sum to the
    GEOMETRIC active return of the reported (net) portfolio over the benchmark."""
    rep = PORTFOLIO_RETURN.validate(portfolio_returns, source="portfolio_returns")
    rp = pd.Series(rep["total_return"].to_numpy(), index=rep["date"].dt.to_period("M")).sort_index()
    rb = _benchmark_monthly(benchmark)
    per = model.per_period[[*model.order, "specific"]]
    return link_multi_period(per, method, portfolio_returns=rp.reindex(per.index), benchmark_returns=rb.reindex(per.index))
