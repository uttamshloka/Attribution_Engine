"""Macro (asset-class exposure) layer of the decision view.

What it measures
----------------
A macro-aware portfolio decides how much equity to hold before it decides
which sectors or stocks. This layer measures the value of that decision
against the STATIC policy equity weight set by the mandate
(``1 - policy.neutral_weights["Cash"]``). The policy weight never moves with
the model; if it did, the layer would measure nothing by construction.

Formula
-------
Per period, Brinson-Fachler over ALL asset classes, funding leg included::

    contribution_t = sum_i (w_p,i,t - w_policy,i) * (r_i,t - r_policy,t)
    r_policy,t     = sum_i w_policy,i * r_i,t

With two asset classes (equity, cash) this reduces exactly to::

    contribution_t = (w_equity,t - w_policy_equity) * (r_equity,t - r_cash,t)

Applying the multi-asset form to the equity line alone drops the funding leg
and silently loses most (with a 100%-equity policy, all) of the effect, so
:func:`asset_class_allocation` refuses weights that leave out any policy
asset class.

The period total is the plain sum of per-period contributions; the
annualised figure is that sum times periods-per-year over the number of
periods. Linking to compounded active return is done later, at the
decision-view level, by the method named in ``meta.linking``.

Which equity return
-------------------
Pass the BENCHMARK's equity return (:func:`benchmark_total_return`). That keeps
sector and stock choices out of this layer: it then prices only the
equity-versus-cash decision. Passing the portfolio's own equity-sleeve return
would mix selection into macro.

Daily regime, monthly returns: the alignment rule
-------------------------------------------------
Each monthly period's equity weight is the TIME-AVERAGE of the daily
``equity_exposure`` over that month's rows, after band clipping. A daily row
describes that day, so the average is the fraction of the month spent in
equity. This is also how the sample portfolio's reported returns are built,
so this layer and the reported returns agree. It is exact when daily
returns are flat within the month and an approximation otherwise (a cut
just before the worst week is credited the same as a cut just after it);
daily-frequency returns would remove that approximation. Every return month
must have regime rows, or the layer fails loudly naming the missing months.

Exactly zero means exactly zero
-------------------------------
Days at policy have an active weight of exactly 0.0 (differences below 1e-12
are snapped to zero), so a month that never leaves policy contributes exactly
0.0, never a small negative from floating-point noise.

Negative is a valid answer
--------------------------
De-risking over a window dominated by a bull market usually costs return.
That negative number is correct and is never clamped or hidden; every report
of it carries the risk metrics beside it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from typing import Literal

import numpy as np
import pandas as pd

from attribution.result import MacroAnatomy
from attribution.schemas import (
    BENCHMARK,
    CASH_KEY,
    CASH_RETURN,
    REGIME_SERIES,
    REGIME_STATES,
    Policy,
)
from attribution.stats import BootstrapResult, block_bootstrap

__all__ = [
    "EQUITY_KEY",
    "INTEGER_STATE_CODES",
    "ZERO_SNAP",
    "MacroLayerError",
    "ClipEvent",
    "Drawdown",
    "Episode",
    "PhaseTotals",
    "RiskMetrics",
    "MacroResult",
    "normalise_regime",
    "benchmark_total_return",
    "equity_band",
    "asset_class_allocation",
    "max_drawdown",
    "capture_ratio",
    "compute_macro_contribution",
]

EQUITY_KEY = "Equity"
#: Integer codes accepted for regime states, mapped to the string names.
INTEGER_STATE_CODES: dict[int, str] = {0: "normal", 1: "elevated", 2: "crisis"}
#: Active weights smaller than this in absolute value are treated as exactly zero.
ZERO_SNAP = 1e-12
_PERIODS_PER_YEAR = 12
_WEIGHT_TOL = 1e-6

Phase = Literal["at_policy", "earned_falling", "given_back_recovering", "off_policy_outside_episode"]


class MacroLayerError(ValueError):
    """Raised when inputs to the macro layer are inconsistent."""


# ---------------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------------


def _normalise_states(values: pd.Series, column: str) -> pd.Series:
    out: list[object] = []
    bad: list[str] = []
    for pos, v in enumerate(values.tolist()):
        if isinstance(v, (bool, np.bool_)):
            bad.append(f"row {pos + 1}: {v!r}")
            out.append(v)
        elif isinstance(v, (int, np.integer)):
            name = INTEGER_STATE_CODES.get(int(v))
            if name is None:
                bad.append(f"row {pos + 1}: {v!r}")
            out.append(name if name is not None else v)
        else:
            out.append(v)
    if bad:
        shown = ", ".join(bad[:10]) + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        codes = ", ".join(f"{k}={s}" for k, s in INTEGER_STATE_CODES.items())
        raise MacroLayerError(f"regime column '{column}': unknown integer state code(s) {shown}; allowed {codes}")
    return pd.Series(out, index=values.index, dtype=object)


def normalise_regime(regime: pd.DataFrame, source: str = "regime") -> pd.DataFrame:
    """Accept string states (``normal``/``elevated``/``crisis``) or integer codes 0/1/2.

    Integer codes are mapped to the strings, then the frame is validated
    against the RegimeSeries contract. Anything else fails loudly.
    """
    df = regime.copy()
    for column in ("raw_state", "confirmed_state"):
        if column in df.columns:
            df[column] = _normalise_states(df[column], column)
    return REGIME_SERIES.validate(df, source=source)


def _as_return_series(data: pd.DataFrame | pd.Series, name: str) -> pd.Series:
    """Validate a return input and return it indexed by monthly period."""
    if isinstance(data, pd.Series):
        frame = pd.DataFrame({"date": list(data.index), "total_return": data.to_numpy()})
    elif isinstance(data, pd.DataFrame):
        frame = data
    else:
        raise TypeError(f"{name}: expected a DataFrame with date,total_return or a Series indexed by date")
    clean = CASH_RETURN.validate(frame, source=name)
    periods = clean["date"].dt.to_period("M")
    dup = periods[periods.duplicated(keep=False)]
    if not dup.empty:
        raise MacroLayerError(f"{name}: more than one row in month(s) {sorted({str(p) for p in dup})}; supply monthly returns")
    return pd.Series(clean["total_return"].to_numpy(), index=pd.PeriodIndex(periods, freq="M"), name=name).sort_index()


def benchmark_total_return(benchmark: pd.DataFrame) -> pd.Series:
    """Benchmark return per date: ``sum(weight * total_return)`` over sectors."""
    clean = BENCHMARK.validate(benchmark, source="benchmark")
    out = (clean["weight"] * clean["total_return"]).groupby(clean["date"]).sum()
    out.name = "benchmark_return"
    return out


def equity_band(policy: Policy) -> tuple[float, float]:
    """Permitted equity range implied by the policy's cash band: ``(1 - cash_high, 1 - cash_low)``."""
    if CASH_KEY in policy.bands:
        lo_cash, hi_cash = policy.bands[CASH_KEY]
        return (1.0 - float(hi_cash), 1.0 - float(lo_cash))
    neutral = policy.neutral_equity_exposure
    return (neutral, neutral)


# ---------------------------------------------------------------------------
# Brinson-Fachler over asset classes
# ---------------------------------------------------------------------------


def asset_class_allocation(
    portfolio_weights: pd.DataFrame,
    policy_weights: Mapping[str, float],
    asset_returns: pd.DataFrame,
) -> pd.Series:
    """Per-period ``sum_i (w_p,i - w_policy,i) * (r_i - r_policy)`` over ALL asset classes.

    ``portfolio_weights`` and ``asset_returns`` are indexed by period with one
    column per asset class. Every policy asset class, including the funding
    leg (cash), must be present and portfolio weights must sum to 1 in every
    period; leaving the funding leg out is refused, because summing the
    equity line alone drops most of the effect.
    """
    policy = {str(k): float(v) for k, v in policy_weights.items()}
    if abs(sum(policy.values()) - 1.0) > _WEIGHT_TOL:
        raise MacroLayerError(f"policy weights sum to {sum(policy.values()):.9f}, must be 1.0")
    for label, frame in (("portfolio_weights", portfolio_weights), ("asset_returns", asset_returns)):
        missing = sorted(set(policy) - set(frame.columns))
        extra = sorted(set(frame.columns) - set(policy))
        if missing:
            raise MacroLayerError(
                f"{label} is missing asset class(es) {missing}. Every leg, including the funding leg, "
                "must be included: summing over the equity line alone drops most of the effect."
            )
        if extra:
            raise MacroLayerError(f"{label} has asset class(es) not in the policy: {extra}")
    if not portfolio_weights.index.equals(asset_returns.index):
        raise MacroLayerError("portfolio_weights and asset_returns must have the same periods")
    cols = list(policy)
    w = portfolio_weights[cols].to_numpy(dtype=float)
    r = asset_returns[cols].to_numpy(dtype=float)
    sums = w.sum(axis=1)
    bad = np.flatnonzero(np.abs(sums - 1.0) > _WEIGHT_TOL)
    if bad.size:
        shown = ", ".join(f"{portfolio_weights.index[i]} sums to {sums[i]:.9f}" for i in bad[:5])
        raise MacroLayerError(
            f"portfolio weights must sum to 1 in every period: {shown}. If the funding leg (cash) is "
            "missing, include it: summing over the equity line alone drops most of the effect."
        )
    wp = np.array([policy[c] for c in cols])
    r_policy = r @ wp
    active = w - wp
    active[np.abs(active) < ZERO_SNAP] = 0.0
    contrib = (active * (r - r_policy[:, None])).sum(axis=1)
    return pd.Series(contrib, index=portfolio_weights.index, name="contribution")


# ---------------------------------------------------------------------------
# Risk helpers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Drawdown:
    """Worst peak-to-trough fall of a wealth index built from period returns.

    ``duration_periods`` runs peak to trough. ``recovery_periods`` runs trough
    back to the old peak, or is ``None`` if the book had not recovered by the
    end of the data. Periods are labelled ``YYYY-MM``; ``start`` means the
    level before the first period.
    """

    max_drawdown: float
    peak: str
    trough: str
    duration_periods: int
    recovery_periods: int | None
    recovered_at: str | None


def max_drawdown(returns: pd.Series) -> Drawdown:
    r = returns.to_numpy(dtype=float)
    labels = ["start"] + [str(p) for p in returns.index]
    wealth = np.concatenate([[1.0], np.cumprod(1.0 + r)])
    running_max = np.maximum.accumulate(wealth)
    dd = wealth / running_max - 1.0
    trough = int(np.argmin(dd))
    if dd[trough] == 0.0:
        return Drawdown(0.0, labels[0], labels[0], 0, 0, labels[0])
    peak = int(np.argmax(wealth[: trough + 1]))
    after = np.flatnonzero(wealth[trough + 1 :] >= wealth[peak])
    if after.size:
        rec = trough + 1 + int(after[0])
        return Drawdown(float(dd[trough]), labels[peak], labels[trough], trough - peak, rec - trough, labels[rec])
    return Drawdown(float(dd[trough]), labels[peak], labels[trough], trough - peak, None, None)


def capture_ratio(book: pd.Series, benchmark: pd.Series, side: Literal["up", "down"]) -> float | None:
    """Geometric-mean capture: the book's mean return in benchmark up (or down) periods
    divided by the benchmark's, both as geometric means. ``None`` when there are
    no such periods."""
    b = benchmark.to_numpy(dtype=float)
    p = book.to_numpy(dtype=float)
    mask = b > 0 if side == "up" else b < 0
    if not mask.any():
        return None
    gb = float(np.prod(1.0 + b[mask]) ** (1.0 / mask.sum()) - 1.0)
    gp = float(np.prod(1.0 + p[mask]) ** (1.0 / mask.sum()) - 1.0)
    return gp / gb


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClipEvent:
    """One daily exposure outside the policy band, and what it was clipped to."""

    date: date
    requested: float
    applied: float
    bound: Literal["lower", "upper"]


@dataclass(frozen=True)
class Episode:
    """A run of consecutive periods with equity below policy (de-risked).

    The turn is the benchmark's trough within the episode: the lowest point of
    the benchmark's cumulative path from the episode start. Periods up to and
    including the trough are the falling phase; later periods are the
    recovering phase. If the benchmark never fell below its starting level,
    there is no trough and every period counts as recovering.
    """

    start: str
    end: str
    n_periods: int
    n_derisked_days: int
    trough: str | None
    average_equity_weight: float
    earned_falling: float
    given_back_recovering: float
    net: float
    states: Mapping[str, int]


@dataclass(frozen=True)
class PhaseTotals:
    earned_falling: float
    given_back_recovering: float
    net_episodes: float
    off_policy_outside_episodes: float


@dataclass(frozen=True)
class RiskMetrics:
    """Risk of the book WITH the overlay versus the same book held at policy."""

    upside_capture: float | None
    downside_capture: float | None
    policy_upside_capture: float | None
    policy_downside_capture: float | None
    overlay_drawdown: Drawdown
    policy_drawdown: Drawdown
    drawdown_difference: float
    share_periods_at_policy: float
    n_episodes: int
    episode_lengths_periods: tuple[int, ...]
    episode_lengths_days: tuple[int, ...]


@dataclass(frozen=True)
class MacroResult:
    mode: str
    applicable: bool
    message: str
    policy_equity_weight: float
    equity_band: tuple[float, float]
    per_period: pd.DataFrame = field(repr=False)
    total: float
    annualised: float
    periods_per_year: int
    clip_events: tuple[ClipEvent, ...]
    episodes: tuple[Episode, ...]
    phases: PhaseTotals
    risk: RiskMetrics
    ci: BootstrapResult | None
    alignment: str

    @property
    def contribution(self) -> pd.Series:
        return self.per_period["contribution"]

    def to_macro_anatomy(self, cash_return_annualised: float) -> MacroAnatomy:
        """The exposure part of ``decision_view.macro_anatomy``. Fees and residual are
        added when the full decision view is assembled."""
        return MacroAnatomy(
            average_equity_exposure=float(self.per_period["equity_weight"].mean()),
            policy_equity_exposure=self.policy_equity_weight,
            cash_return=cash_return_annualised,
            components={"exposure": self.annualised},
        )

    def report(self) -> str:
        """Human-readable summary. The contribution is never shown without its risk metrics."""

        def pct(x: float | None, digits: int = 2) -> str:
            return "n/a" if x is None else f"{x * 100:+.{digits}f}%"

        def ratio(x: float | None) -> str:
            return "n/a" if x is None else f"{x:.3f}"

        def dd_line(label: str, d: Drawdown) -> str:
            rec = f"recovered after {d.recovery_periods} period(s) ({d.recovered_at})" if d.recovery_periods is not None else "NOT recovered by end of data"
            return (f"  max drawdown {label:15s} {pct(d.max_drawdown)}  peak {d.peak} -> trough {d.trough} "
                    f"({d.duration_periods} period(s)), {rec}")

        r = self.risk
        lines = [
            "MACRO (ASSET-CLASS EXPOSURE) LAYER",
            f"  derisk_mode           {self.mode}",
            f"  {self.message}",
            f"  policy equity weight  {self.policy_equity_weight:.4f} (static, from the mandate)",
            f"  permitted equity band {self.equity_band[0]:.4f} to {self.equity_band[1]:.4f}",
            f"  periods               {len(self.per_period)} ({self.per_period.index[0]} to {self.per_period.index[-1]})",
            "",
            "CONTRIBUTION",
            f"  total (sum of periods)   {pct(self.total, 3)}",
            f"  annualised               {pct(self.annualised, 3)} a year",
        ]
        if self.ci is not None:
            lines.append(f"  95% interval (annualised) {self.ci.describe(scale=100, unit='%')}")
        if self.total < 0:
            lines.append("  NOTE: negative is a valid result - de-risking usually costs return in rising markets. "
                         "Judge it with the risk metrics below.")
        lines += [
            "",
            "TWO-PHASE SPLIT (within de-risked episodes)",
            f"  earned while benchmark falling        {pct(self.phases.earned_falling, 3)}",
            f"  given back while benchmark recovering {pct(self.phases.given_back_recovering, 3)}",
            f"  net over episodes                     {pct(self.phases.net_episodes, 3)}",
            f"  off policy outside episodes           {pct(self.phases.off_policy_outside_episodes, 3)}",
        ]
        for i, e in enumerate(self.episodes, 1):
            lines.append(
                f"  episode {i}: {e.start} to {e.end}, {e.n_periods} period(s), {e.n_derisked_days} de-risked day(s), "
                f"trough {e.trough or 'none'}, avg equity {e.average_equity_weight:.3f}: "
                f"earned {pct(e.earned_falling, 3)}, given back {pct(e.given_back_recovering, 3)}, net {pct(e.net, 3)}"
            )
        lines += [
            "",
            "RISK METRICS (book with overlay vs same book held at policy)",
            f"  upside capture    overlay {ratio(r.upside_capture)}   policy {ratio(r.policy_upside_capture)}",
            f"  downside capture  overlay {ratio(r.downside_capture)}   policy {ratio(r.policy_downside_capture)}",
            dd_line("with overlay", r.overlay_drawdown),
            dd_line("at policy", r.policy_drawdown),
            f"  drawdown difference (overlay - policy) {pct(r.drawdown_difference)}  (positive = shallower with overlay)",
            f"  share of periods at policy weight {r.share_periods_at_policy * 100:.1f}% (contribution exactly zero)",
            f"  de-risking episodes {r.n_episodes}; lengths {list(r.episode_lengths_periods)} period(s), "
            f"{list(r.episode_lengths_days)} day(s)",
            "",
            f"  clipping: {len(self.clip_events)} daily exposure(s) outside the band were clipped",
            f"  alignment: {self.alignment}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# The layer
# ---------------------------------------------------------------------------

_ALIGNMENT = ("monthly equity weight = time-average of daily equity_exposure (after band clipping) over "
              "the regime rows in that calendar month")


def compute_macro_contribution(
    regime: pd.DataFrame,
    policy: Policy,
    equity_returns: pd.DataFrame | pd.Series,
    cash_returns: pd.DataFrame | pd.Series,
    benchmark_returns: pd.DataFrame | pd.Series | None = None,
    *,
    bootstrap: bool = True,
    seed: int = 20261028,
) -> MacroResult:
    """Value of the equity-versus-cash decision against the static policy weight.

    ``regime``: daily (or monthly) RegimeSeries; string or integer-coded states.
    ``equity_returns``: monthly equity return to price the decision with; use
    :func:`benchmark_total_return` so selection stays out of this layer.
    ``cash_returns``: monthly cash return (the funding leg).
    ``benchmark_returns``: series used to find the trough and for capture
    ratios; defaults to ``equity_returns``.
    """
    reg = normalise_regime(regime)
    r_eq = _as_return_series(equity_returns, "equity_returns")
    r_cash = _as_return_series(cash_returns, "cash_returns")
    r_bench = r_eq if benchmark_returns is None else _as_return_series(benchmark_returns, "benchmark_returns")

    if not r_eq.index.equals(r_cash.index):
        only_eq = sorted(str(p) for p in r_eq.index.difference(r_cash.index))
        only_cash = sorted(str(p) for p in r_cash.index.difference(r_eq.index))
        raise MacroLayerError(f"equity and cash returns cover different months: only equity {only_eq}, only cash {only_cash}")
    if not r_bench.index.equals(r_eq.index):
        raise MacroLayerError("benchmark_returns must cover the same months as equity_returns")
    months = r_eq.index

    wp = policy.neutral_equity_exposure
    lo, hi = equity_band(policy)
    mode = policy.derisk_mode

    # Daily weights, restricted to the return months.
    reg = reg.assign(period=reg["date"].dt.to_period("M"))
    reg = reg[reg["period"].isin(months)]
    missing = sorted(str(p) for p in months.difference(pd.PeriodIndex(reg["period"].unique(), freq="M")))
    if missing:
        raise MacroLayerError(f"regime has no rows for return month(s) {missing}")

    clip_events: list[ClipEvent] = []
    if mode == "cash":
        requested = reg["equity_exposure"].to_numpy(dtype=float)
        applied = np.clip(requested, lo, hi)
        for d, q, a in zip(reg["date"], requested, applied, strict=True):
            if q < lo:
                clip_events.append(ClipEvent(d.date(), float(q), float(a), "lower"))
            elif q > hi:
                clip_events.append(ClipEvent(d.date(), float(q), float(a), "upper"))
        active_daily = applied - wp
        active_daily[np.abs(active_daily) < ZERO_SNAP] = 0.0
        message = ("Equity exposure follows the regime model's equity_exposure, clipped to the policy band; "
                   "measured against the static policy weight.")
    elif mode == "within_equity":
        active_daily = np.zeros(len(reg))
        off = int((np.abs(reg["equity_exposure"].to_numpy(dtype=float) - wp) >= ZERO_SNAP).sum())
        message = (f"NOT APPLICABLE - derisk_mode is 'within_equity': equity is held at the policy weight and the "
                   f"defensive expression happens in the sector tilt, so this layer is exactly zero by construction. "
                   f"({off} daily regime exposure value(s) differ from policy and were ignored here.)")
    else:  # Policy already validates this; kept so a future mode cannot pass silently.
        raise MacroLayerError(f"unsupported derisk_mode {mode!r}")

    reg = reg.assign(active=active_daily)
    active_m = reg.groupby("period")["active"].mean().reindex(months)
    active_m[np.abs(active_m) < ZERO_SNAP] = 0.0
    derisked_days = reg.assign(d=reg["active"] < 0).groupby("period")["d"].sum().reindex(months).astype(int)
    states_m = reg.groupby("period")["confirmed_state"].agg(lambda s: s.value_counts().to_dict()).reindex(months)
    w_eq = wp + active_m

    # Brinson-Fachler over both legs; equals the two-asset shorthand exactly.
    weights = pd.DataFrame({EQUITY_KEY: w_eq, CASH_KEY: 1.0 - w_eq}, index=months)
    returns = pd.DataFrame({EQUITY_KEY: r_eq, CASH_KEY: r_cash}, index=months)
    contrib = asset_class_allocation(weights, {EQUITY_KEY: wp, CASH_KEY: 1.0 - wp}, returns)

    # Episodes and the two-phase split.
    derisked = (active_m < 0).to_numpy()
    phase = np.where(active_m.to_numpy() == 0.0, "at_policy", "off_policy_outside_episode").astype(object)
    episodes: list[Episode] = []
    i, n = 0, len(months)
    while i < n:
        if not derisked[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and derisked[j + 1]:
            j += 1
        seg = slice(i, j + 1)
        path = np.concatenate([[1.0], np.cumprod(1.0 + r_bench.to_numpy()[seg])])
        k = int(np.argmin(path))  # 0 = start level, i.e. never fell
        trough_pos = i + k - 1 if k > 0 else None
        for t in range(i, j + 1):
            phase[t] = "earned_falling" if trough_pos is not None and t <= trough_pos else "given_back_recovering"
        c = contrib.to_numpy()[seg]
        ph = phase[seg]
        earned = float(c[ph == "earned_falling"].sum())
        given = float(c[ph == "given_back_recovering"].sum())
        counts: dict[str, int] = {}
        for m in months[seg]:
            for s, v in states_m[m].items():
                counts[s] = counts.get(s, 0) + int(v)
        episodes.append(Episode(
            start=str(months[i]), end=str(months[j]), n_periods=j - i + 1,
            n_derisked_days=int(derisked_days.iloc[seg].sum()),
            trough=str(months[trough_pos]) if trough_pos is not None else None,
            average_equity_weight=float(w_eq.iloc[seg].mean()),
            earned_falling=earned, given_back_recovering=given, net=earned + given,
            states={s: counts[s] for s in REGIME_STATES if s in counts},
        ))
        i = j + 1

    in_episode = np.isin(phase, ["earned_falling", "given_back_recovering"])
    c_all = contrib.to_numpy()
    phases = PhaseTotals(
        earned_falling=float(c_all[phase == "earned_falling"].sum()),
        given_back_recovering=float(c_all[phase == "given_back_recovering"].sum()),
        net_episodes=float(c_all[in_episode].sum()),
        off_policy_outside_episodes=float(c_all[phase == "off_policy_outside_episode"].sum()),
    )

    # Books: with the overlay, and the same book held at policy.
    overlay_book = w_eq * r_eq + (1.0 - w_eq) * r_cash
    policy_book = wp * r_eq + (1.0 - wp) * r_cash
    dd_overlay, dd_policy = max_drawdown(overlay_book), max_drawdown(policy_book)
    risk = RiskMetrics(
        upside_capture=capture_ratio(overlay_book, r_bench, "up"),
        downside_capture=capture_ratio(overlay_book, r_bench, "down"),
        policy_upside_capture=capture_ratio(policy_book, r_bench, "up"),
        policy_downside_capture=capture_ratio(policy_book, r_bench, "down"),
        overlay_drawdown=dd_overlay,
        policy_drawdown=dd_policy,
        drawdown_difference=dd_overlay.max_drawdown - dd_policy.max_drawdown,
        share_periods_at_policy=float((contrib.to_numpy() == 0.0).mean()),
        n_episodes=len(episodes),
        episode_lengths_periods=tuple(e.n_periods for e in episodes),
        episode_lengths_days=tuple(e.n_derisked_days for e in episodes),
    )

    total = float(contrib.sum())
    annualised = total * _PERIODS_PER_YEAR / len(months)
    ci = None
    if bootstrap and mode == "cash" and len(months) >= 2:
        ci = block_bootstrap(contrib.to_numpy(), lambda a: float(a.mean()) * _PERIODS_PER_YEAR,
                             frequency="monthly", seed=seed)

    per_period = pd.DataFrame({
        "equity_weight": w_eq,
        "policy_weight": wp,
        "active_weight": active_m,
        "derisked_days": derisked_days,
        "equity_return": r_eq,
        "cash_return": r_cash,
        "benchmark_return": r_bench,
        "contribution": contrib,
        "phase": phase,
    }, index=months)
    per_period.index.name = "period"

    if math.isnan(total):
        raise MacroLayerError("contribution is NaN - check the inputs")
    return MacroResult(
        mode=mode, applicable=mode == "cash", message=message, policy_equity_weight=wp,
        equity_band=(lo, hi), per_period=per_period, total=total, annualised=annualised,
        periods_per_year=_PERIODS_PER_YEAR, clip_events=tuple(clip_events), episodes=tuple(episodes),
        phases=phases, risk=risk, ci=ci, alignment=_ALIGNMENT,
    )
