"""One entry point for the whole engine: :func:`run_attribution`.

It loads the inputs from a folder, runs the macro layer, Brinson and the
factor model, assembles and validates the :class:`attribution.outputs.Result`,
and returns it. The app never needs to call those modules directly.

Which return series each view is built on
------------------------------------------
Write ``r_gross`` for the holdings-based portfolio return (holdings x prices,
blended with cash at the equity weight), ``r_net`` for the reported return
(net of fees) and ``r_policy = w_pol * r_benchmark + (1 - w_pol) * r_cash``
for the policy benchmark (the benchmark itself when the policy is 100% equity).

* DECISION VIEW, four layers - macro (exposure timing only), allocation,
  selection, interaction - built on ``r_gross`` vs ``r_policy`` and linked
  with coefficients computed on that GROSS pair. They sum to

      gross active = prod(1 + r_gross) - prod(1 + r_policy)

* COST LINE - fees and trading costs, in the same cumulative space:

      cost = prod(1 + r_net) - prod(1 + r_gross)
      gross active + cost = net active

* FACTOR VIEW - built on ``r_net`` vs ``r_policy``, linked on that NET pair.
  It sums to

      net active = prod(1 + r_net) - prod(1 + r_policy)

So the decision view spans gross and net (four layers, then the cost line),
and the factor view is net throughout. The gate checks each identity
separately.

Inputs (CSV or Parquet, by these names): holdings, benchmark, prices,
portfolio_returns, factor_returns, cash_returns, regime_series; policy.json
unless a Policy is passed.
"""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from attribution import __version__, io
from attribution.brinson import EFFECTS, link_multi_period, run_decision_attribution
from attribution.factors import FACTOR_ORDER, MARKET_VS_MACRO_NOTE, TIMING_TERM_NOTE, build_factor_matrix, fit_factor_model
from attribution.macro_layer import equity_band
from attribution.outputs import GrossToNet, InputChecks, Limitation, ReconciliationDetail, Result, RunInfo
from attribution.result import (
    AttributionResult,
    ConfidenceInterval,
    DecisionLayer,
    DecisionView,
    FactorContribution,
    FactorView,
    MacroAnatomy,
    Meta,
    Period,
    Reconciliation,
    SectorAttribution,
    SpecificReturn,
    Totals,
)
from attribution.schemas import FACTORS, SECTORS, Policy
from attribution.stats import BootstrapResult, block_bootstrap
from attribution.validate import check_min_months, validate_result

__all__ = ["INPUT_STEMS", "EngineConfig", "EngineError", "run_attribution"]

INPUT_STEMS: tuple[str, ...] = (
    "holdings", "benchmark", "prices", "portfolio_returns", "factor_returns", "cash_returns", "regime_series",
)
_LOADERS = {
    "holdings": io.load_holdings,
    "benchmark": io.load_benchmark,
    "prices": io.load_prices,
    "portfolio_returns": io.load_portfolio_returns,
    "factor_returns": io.load_factor_returns,
    "cash_returns": io.load_cash_returns,
    "regime_series": io.load_regime_series,
}
_PPY = 12


class EngineError(ValueError):
    """Inputs missing or unusable before attribution could start."""


@dataclass(frozen=True)
class EngineConfig:
    portfolio_name: str = "Portfolio"
    portfolio_id: str = ""
    currency: str = "INR"
    linking: str = "carino"
    brinson_variant: str = "fachler"
    factor_order: tuple[str, ...] = FACTOR_ORDER
    bootstrap: bool = True
    seed: int = 20261028
    tolerance: float = 1e-9
    gross_net_threshold_monthly: float = 0.005
    short_history_months: int = 36
    min_episodes: int = 3
    generated_at_utc: datetime | None = field(default=None)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def _find(data_dir: Path, stem: str) -> Path:
    for suffix in (".csv", ".parquet"):
        p = data_dir / f"{stem}{suffix}"
        if p.exists():
            return p
    raise EngineError(f"input '{stem}' not found in {data_dir} (looked for {stem}.csv and {stem}.parquet)")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _restrict(frames: dict[str, pd.DataFrame], period: tuple[str | date, str | date] | None) -> dict[str, pd.DataFrame]:
    if period is None:
        return frames
    start = pd.Period(pd.Timestamp(period[0]), freq="M")
    end = pd.Period(pd.Timestamp(period[1]), freq="M")
    if end < start:
        raise EngineError(f"period end {end} is before start {start}")
    out = {}
    for name, df in frames.items():
        per = df["date"].dt.to_period("M")
        out[name] = df.loc[(per >= start) & (per <= end)].reset_index(drop=True)
        if out[name].empty:
            raise EngineError(f"input '{name}' has no rows between {start} and {end}")
    return out


def _is_synthetic(paths: Sequence[Path]) -> bool:
    for p in paths:
        try:
            if any("SYNTHETIC" in line.upper() for line in io.read_header_comment(p)):
                return True
        except (OSError, ValueError):
            continue
    return False


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ann(cumulative: float, n: int) -> float:
    return (1.0 + cumulative) ** (_PPY / n) - 1.0


def _series(df: pd.DataFrame) -> pd.Series:
    return pd.Series(df["total_return"].to_numpy(), index=df["date"].dt.to_period("M")).sort_index()


def _ci_annual_mean(values: np.ndarray, seed: int) -> BootstrapResult:
    return block_bootstrap(values, lambda a: float(a.mean()) * _PPY, frequency="monthly", seed=seed)


def _sector_pairs(df: pd.DataFrame) -> set[str]:
    live = df.loc[df["weight"] > 0]
    return {f"{p} {s}" for p, s in zip(live["date"].dt.to_period("M"), live["sector"], strict=True)}


# ---------------------------------------------------------------------------
# The entry point
# ---------------------------------------------------------------------------


def run_attribution(
    data_dir: str | Path,
    policy: Policy | str | Path | None = None,
    period: tuple[str | date, str | date] | None = None,
    config: EngineConfig | None = None,
) -> Result:
    """Run the full engine on the files in ``data_dir`` and return a validated :class:`Result`.

    ``policy``: a Policy, a path to a policy JSON, or None to read
    ``data_dir/policy.json``. ``period``: optional (start, end) months,
    inclusive, e.g. ``("2022-01", "2026-09")``. Raises
    :class:`attribution.validate.GateError` if the result fails validation.
    """
    cfg = config or EngineConfig()
    root = Path(data_dir)
    if not root.is_dir():
        raise EngineError(f"data_dir {root} is not a folder")
    paths = {stem: _find(root, stem) for stem in INPUT_STEMS}
    if policy is None:
        policy_path: Path | None = root / "policy.json"
        if not policy_path.exists():
            raise EngineError(f"no policy given and {policy_path} does not exist")
        pol = io.load_policy(policy_path)
    elif isinstance(policy, Policy):
        policy_path, pol = None, policy
    else:
        policy_path = Path(policy)
        pol = io.load_policy(policy_path)

    frames = {stem: _LOADERS[stem](p) for stem, p in paths.items()}
    frames = _restrict(frames, period)
    holdings, benchmark, prices = frames["holdings"], frames["benchmark"], frames["prices"]
    reported, factor_returns, cash = frames["portfolio_returns"], frames["factor_returns"], frames["cash_returns"]
    regime = frames["regime_series"]

    months = sorted(set(holdings["date"].dt.to_period("M")))
    check_min_months(len(months))
    n = len(months)

    # --- decision view: four layers on GROSS vs policy ----------------------------
    dec = run_decision_attribution(holdings, benchmark, prices, regime, pol, cash, reported,
                                   variant=cfg.brinson_variant, method=cfg.linking,  # type: ignore[arg-type]
                                   bootstrap=cfg.bootstrap, seed=cfg.seed)
    pp = dec.per_period
    r_net, r_gross, r_pol = pp["r_portfolio_reported"], pp["r_portfolio_gross"], pp["r_policy"]
    layers = dec.linked.totals  # keys: macro, allocation, selection_effect, interaction (gross-linked)
    sectors = dec.by_sector      # linked with the same gross factors
    period_gap_dec = float(pp["identity_gap"].abs().max())

    # --- factor view: NET vs policy ---------------------------------------------------
    matrix = build_factor_matrix(factor_returns, benchmark, cash)
    active_m = (r_net - r_pol).rename("active")
    model = fit_factor_model(active_m, matrix, order=cfg.factor_order, bootstrap=cfg.bootstrap, seed=cfg.seed)
    fac_cols = model.per_period[[*model.order, "specific"]]
    fac_link = link_multi_period(fac_cols, cfg.linking, portfolio_returns=r_net, benchmark_returns=r_pol)  # type: ignore[arg-type]
    period_gap_fac = float((fac_cols.sum(axis=1) - active_m).abs().max())

    # --- totals and the cost line --------------------------------------------------------
    net_cum = float((1 + r_net).prod() - 1)
    gross_cum = float((1 + r_gross).prod() - 1)
    pol_cum = float((1 + r_pol).prod() - 1)
    active = net_cum - pol_cum
    gross_active = gross_cum - pol_cum
    cost_line = net_cum - gross_cum
    active_ci = _ci_annual_mean(active_m.to_numpy(), cfg.seed)

    layers_total = layers["macro"] + sum(layers[c] for c in EFFECTS)
    decision_total = layers_total + cost_line
    factor_total = fac_link.linked_sum
    tol = cfg.tolerance
    reconciliation = Reconciliation.compute(decision_total, factor_total, active, tolerance=tol,
                                            layers_total=layers_total, gross_active=gross_active)

    # --- gross to net (data-consistency check) ----------------------------------------
    gap = (r_net - r_gross).to_numpy()
    mean_abs_gap = float(np.abs(gap).mean())
    consistent = mean_abs_gap <= cfg.gross_net_threshold_monthly
    gross_to_net = GrossToNet(
        gross_cumulative=gross_cum, net_cumulative=net_cum, cost_cumulative=cost_line,
        gross_annualised=_ann(gross_cum, n), net_annualised=_ann(net_cum, n),
        cost_annualised=_ann(net_cum, n) - _ann(gross_cum, n),
        mean_monthly_gap=float(gap.mean()), mean_abs_monthly_gap=mean_abs_gap,
        max_abs_monthly_gap=float(np.abs(gap).max()), threshold_monthly=cfg.gross_net_threshold_monthly,
        consistent=consistent,
        label="fees and trading costs" if consistent else
              "UNEXPLAINED gap - holdings x prices and reported returns may not describe the same portfolio",
    )

    # --- input checks (evidence for the gate) ---------------------------------
    hw = holdings.groupby(holdings["date"].dt.to_period("M"))["weight"].sum()
    bw = benchmark.groupby(benchmark["date"].dt.to_period("M"))["weight"].sum()
    held, in_bm = _sector_pairs(holdings), _sector_pairs(benchmark)
    lo, hi = equity_band(pol)
    checks = InputChecks(
        n_months=n,
        holdings_weight_max_deviation=float((hw - 1).abs().max()),
        benchmark_weight_max_deviation=float((bw - 1).abs().max()),
        held_not_in_benchmark=tuple(sorted(held - in_bm)),
        benchmark_not_held=tuple(sorted(in_bm - held)),
        orphans_recorded=tuple(sorted(f"{o.period} {o.sector}" for o in dec.effects.orphans)),
        dropped_holdings=tuple(f"{d.date} {d.security_id} ({d.weight:.4%})" for d in dec.effects.dropped),
        policy_equity_weight=pol.neutral_equity_exposure,
        policy_equity_band=(lo, hi),
    )

    # --- the contract-level AttributionResult ----------------------------------
    rows = []
    for s in SECTORS:
        r = sectors.loc[s]
        rp = r["r_p"] if np.isfinite(r["r_p"]) else (r["r_b"] if np.isfinite(r["r_b"]) else 0.0)
        rb = r["r_b"] if np.isfinite(r["r_b"]) else (r["r_p"] if np.isfinite(r["r_p"]) else 0.0)
        rows.append(SectorAttribution(s, float(r["w_p"]), float(r["w_b"]), float(rp), float(rb),
                                      float(r["allocation"]), float(r["selection_effect"]), float(r["interaction"])))
    cash_s = _series(cash)
    decision_view = DecisionView(
        layers=(
            DecisionLayer("macro", layers["macro"]),
            DecisionLayer("allocation", layers["allocation"]),
            DecisionLayer("selection", layers["selection_effect"]),
            DecisionLayer("interaction", layers["interaction"]),
        ),
        by_sector=tuple(rows),
        macro_anatomy=MacroAnatomy(
            average_equity_exposure=float(pp["equity_weight"].mean()),
            policy_equity_exposure=pol.neutral_equity_exposure,
            cash_return=_ann(float((1 + cash_s.reindex(pp.index)).prod() - 1), n),
            components={"exposure": layers["macro"]},  # exposure timing ONLY - never costs
        ),
        cost=cost_line,
    )
    fv_model = model.to_factor_view()
    by_name = {r.factor: r for r in model.rows}
    factor_view = FactorView(
        factors=tuple(
            FactorContribution(f, exposure=by_name[f].exposure, factor_return=by_name[f].factor_mean_annual,
                               contribution=fac_link.totals[f], t_stat=by_name[f].t_stat)
            for f in FACTORS
        ),
        specific_return=SpecificReturn(value=fac_link.totals["specific"], t_stat=model.specific_t_stat,
                                       information_ratio=fv_model.specific_return.information_ratio),
        diagnostics=dataclasses.replace(
            fv_model.diagnostics, method=fv_model.diagnostics.method + f"; contributions {cfg.linking}-linked to geometric active"),
    )

    # --- limitations -----------------------------------------------------------
    lims = _limitations(cfg, n, dec, model, gross_to_net, checks, active_ci,
                        synthetic=_is_synthetic(list(paths.values())))

    generated = cfg.generated_at_utc or datetime.now(timezone.utc).replace(tzinfo=None, microsecond=0)
    attribution = AttributionResult(
        meta=Meta(
            portfolio_id=cfg.portfolio_id or cfg.portfolio_name, portfolio_name=cfg.portfolio_name,
            benchmark_name=pol.benchmark_name, currency=cfg.currency, frequency="monthly",
            linking=cfg.linking, engine_version=__version__,  # type: ignore[arg-type]
            generated_at=generated, synthetic_data=_is_synthetic(list(paths.values())),
        ),
        period=Period(months[0].to_timestamp(how="end").date(), months[-1].to_timestamp(how="end").date(), n),
        totals=Totals(portfolio=net_cum, benchmark=pol_cum, active=active,
                      portfolio_gross=gross_cum, active_gross=gross_active,
                      ci=ConfidenceInterval(level=active_ci.level, lower=active_ci.lower, upper=active_ci.upper,
                                            method=active_ci.method + " of annualised mean monthly active return")),
        decision_view=decision_view,
        factor_view=factor_view,
        reconciliation=reconciliation,
        limitations=tuple(f"[{x.severity}] {x.text}" for x in lims),
    )

    detail = ReconciliationDetail(
        active_definition="net active = prod(1 + portfolio net) - prod(1 + policy benchmark); policy benchmark = "
                          "w_pol x benchmark + (1 - w_pol) x cash",
        series_used={
            "decision_layers": "GROSS portfolio (holdings x prices, cash at equity weight) vs policy benchmark; "
                               f"{cfg.linking} coefficients on that gross pair",
            "cost_line": "prod(1 + NET portfolio) - prod(1 + GROSS portfolio)",
            "factor_view": f"NET portfolio (reported) vs policy benchmark; {cfg.linking} coefficients on that net pair",
        },
        linking=cfg.linking,
        active_return=active,
        gross_active=gross_active,
        decision_layers_total=layers_total,
        cost_line=cost_line,
        decision_total=decision_total,
        factor_total=factor_total,
        layers_minus_gross_active=layers_total - gross_active,
        gross_plus_cost_minus_net=gross_active + cost_line - active,
        decision_minus_active=decision_total - active,
        factor_minus_active=factor_total - active,
        decision_minus_factor=decision_total - factor_total,
        tolerance=tol,
        passes=bool(abs(layers_total - gross_active) <= tol and abs(gross_active + cost_line - active) <= tol
                    and abs(factor_total - active) <= tol and abs(decision_total - factor_total) <= tol),
        decision_components={"macro": layers["macro"], "allocation": layers["allocation"],
                             "selection": layers["selection_effect"], "interaction": layers["interaction"]},
        factor_components={**{f: fac_link.totals[f] for f in model.order}, "specific_return": fac_link.totals["specific"]},
        max_period_identity_gap=max(period_gap_dec, period_gap_fac),
    )
    inputs = {stem: f"{p.name} sha256:{_sha256(p)}" for stem, p in paths.items()}
    if policy_path is not None:
        inputs["policy"] = f"{policy_path.name} sha256:{_sha256(policy_path)}"
    else:
        inputs["policy"] = "Policy object passed in code"
    run = RunInfo(package_version=__version__, linking=cfg.linking, brinson_variant=cfg.brinson_variant,
                  factor_order=tuple(model.order), data_dir=str(root.resolve()), inputs=inputs,
                  generated_at_utc=generated)

    result = Result(attribution=attribution, gross_to_net=gross_to_net, reconciliation=detail,
                    checks=checks, limitations=lims, run=run)
    return validate_result(result)


# ---------------------------------------------------------------------------
# Consolidated limitations
# ---------------------------------------------------------------------------


def _limitations(cfg: EngineConfig, n: int, dec, model, g2n: GrossToNet, checks: InputChecks,
                 active_ci: BootstrapResult, synthetic: bool) -> tuple[Limitation, ...]:
    out: list[Limitation] = []

    def add(id_: str, severity: str, text: str, *affects: str) -> None:
        out.append(Limitation(id_, severity, text, tuple(affects)))  # type: ignore[arg-type]

    if synthetic:
        add("synthetic_data", "material", "Inputs are SYNTHETIC sample data; figures describe no real portfolio.",
            "meta", "totals")
    if not g2n.consistent:
        add("gross_net_inconsistent", "material",
            f"Holdings x prices and reported returns differ by {g2n.mean_abs_monthly_gap:.2%} a month on average "
            f"(max {g2n.max_abs_monthly_gap:.2%}), above the {g2n.threshold_monthly:.2%} threshold: the two inputs may "
            "not describe the same portfolio. The gap is NOT labelled as fees.", "gross_to_net", "decision_view.cost")
    if n < cfg.short_history_months:
        add("short_history", "caution", f"Only {n} months of holdings history (below {cfg.short_history_months}); "
            "estimates, intervals and factor exposures are less reliable.", "decision_view", "factor_view")
    if model.condition_number_raw > 30:
        add("factor_collinearity", "material",
            f"Factor model condition number is {model.condition_number_raw:.1f} (above 30): raw factors are strongly "
            "collinear and individual factor lines are unstable.", "factor_view")
    macro = dec.macro
    if macro.mode == "within_equity":
        add("macro_not_applicable", "info", "derisk_mode is 'within_equity': the macro exposure effect is zero by "
            "construction; defensive positioning shows in sector allocation.", "decision_view.macro")
    elif macro.risk.n_episodes == 0:
        add("macro_never_fired", "caution", "The overlay never left policy in this window: the macro exposure effect "
            "is zero by construction, not by measurement.", "decision_view.macro")
    if macro.mode == "cash" and macro.risk.n_episodes < cfg.min_episodes:
        add("few_derisking_episodes", "caution",
            f"Only {macro.risk.n_episodes} de-risking episode(s) in the window (fewer than {cfg.min_episodes}): "
            "evidence on the overlay rests on very few events.", "decision_view.macro")
    if not cfg.bootstrap:
        add("no_intervals", "caution", "Bootstrap was switched off: per-line confidence intervals were not computed.",
            "decision_view", "factor_view")
    else:
        spans = []
        if macro.ci is not None and macro.ci.spans_zero:
            spans.append(("decision_view.macro", "macro exposure effect", macro.ci))
        for name, label in (("allocation", "allocation"), ("selection_effect", "selection"), ("interaction", "interaction")):
            ci = dec.intervals.get(name)
            if ci is not None and ci.spans_zero:
                spans.append((f"decision_view.{label}", label, ci))
        for r in model.rows:
            if r.ci is not None and r.ci.spans_zero:
                spans.append((f"factor_view.{r.factor}", f"{r.factor} factor contribution", r.ci))
        if model.specific_ci is not None and model.specific_ci.spans_zero:
            spans.append(("factor_view.specific_return", "specific return", model.specific_ci))
        for affects, label, ci in spans:
            add(f"ci_spans_zero:{affects}", "caution",
                f"The 95% interval for the {label} spans zero (annualised {ci.lower:+.2%} to {ci.upper:+.2%}): "
                "not statistically distinguishable from zero.", affects)
    if active_ci.spans_zero:
        add("ci_spans_zero:totals.active", "caution",
            f"The 95% interval for annualised active return spans zero ({active_ci.lower:+.2%} to {active_ci.upper:+.2%}).",
            "totals.active")
    if checks.benchmark_not_held or checks.held_not_in_benchmark:
        sectors_out = sorted({x.split(" ", 1)[1] for x in checks.benchmark_not_held})
        sectors_in = sorted({x.split(" ", 1)[1] for x in checks.held_not_in_benchmark})
        text = "Orphan sectors recorded:"
        if sectors_out:
            text += f" in benchmark but not held - {', '.join(sectors_out)} (selection and interaction zero, " \
                    "portfolio return shown as the benchmark sector return);"
        if sectors_in:
            text += f" held but not in benchmark - {', '.join(sectors_in)} (whole bet counted as allocation);"
        add("orphan_sectors", "info", text.rstrip(";") + ".", "decision_view.by_sector")
    if checks.dropped_holdings:
        add("dropped_holdings", "caution", f"{len(checks.dropped_holdings)} holding(s) had no price and were excluded "
            "(weights rescaled, never imputed): " + "; ".join(checks.dropped_holdings[:5]), "decision_view")
    for i, text in enumerate(model.limitations):
        if "Condition number" in text:
            continue  # reported above as factor_collinearity
        add(f"factor_note_{i + 1}", "info", text, "factor_view")
    add("market_vs_macro", "info", MARKET_VS_MACRO_NOTE, "factor_view.market", "decision_view.macro")
    add("timing_term_not_implemented", "info", TIMING_TERM_NOTE, "factor_view.market", "decision_view.macro")
    add("arithmetic_intervals", "info", "Confidence intervals are on annualised arithmetic means of monthly "
        "contributions; headline lines are geometrically linked cumulative figures.", "decision_view", "factor_view")
    return tuple(out)
