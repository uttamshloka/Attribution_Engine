"""Properties of the holdings-based Brinson attribution.

Properties, not numbers, so the tests survive regenerating the sample data.
"""

from __future__ import annotations

import dataclasses
import math
import pathlib
import re

import numpy as np
import pandas as pd
import pytest

from attribution.brinson import (
    EFFECTS,
    BrinsonError,
    DecisionAttribution,
    link_multi_period,
    linking_factors,
    run_decision_attribution,
    scale_by_exposure,
    single_period_effects,
)
from attribution.sample import SampleData
from attribution.schemas import CASH_KEY, SECTORS

TOL = 1e-12


def _run(sample: SampleData, **kw: object) -> DecisionAttribution:
    args: dict[str, object] = dict(
        holdings=sample.holdings, benchmark=sample.benchmark, prices=sample.prices, regime=sample.regime,
        policy=sample.policy, cash_returns=sample.cash_returns, portfolio_returns=sample.portfolio_returns,
        bootstrap=False,
    )
    args.update(kw)
    return run_decision_attribution(**args)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def decision(sample: SampleData) -> DecisionAttribution:
    return run_decision_attribution(
        sample.holdings, sample.benchmark, sample.prices, sample.regime, sample.policy,
        sample.cash_returns, sample.portfolio_returns,
    )


def _toy(dates: list[str], bench: dict[str, list[tuple[float, float]]], hold: dict[str, list[tuple[str, float, float]]]):
    """Tiny world. bench[sector] = [(weight, return) per date];
    hold[sector] = [(security_id, weight, return) per date] (weight 0 = not held that date)."""
    b_rows, h_rows, p_rows = [], [], []
    for t, d in enumerate(dates):
        for s, vals in bench.items():
            b_rows.append({"date": d, "sector": s, "weight": vals[t][0], "total_return": vals[t][1]})
        for s, vals in hold.items():
            sid, w, r = vals[t]
            if w > 0:
                h_rows.append({"date": d, "security_id": sid, "weight": w, "sector": s})
            p_rows.append({"date": d, "security_id": sid, "total_return": r})
    return pd.DataFrame(h_rows), pd.DataFrame(b_rows), pd.DataFrame(p_rows)


# --- the core identities -----------------------------------------------------


def test_effects_sum_to_equity_book_active_every_period(decision: DecisionAttribution) -> None:
    pp = decision.effects.per_period()
    gap = (pp[list(EFFECTS)].sum(axis=1) - pp["active_equity_book"]).abs()
    assert gap.max() <= TOL


def test_joint_identity_with_macro_layer_every_period(decision: DecisionAttribution) -> None:
    """r_P - r_policy = macro + w_eq * sum(effects), period by period."""
    f = decision.per_period
    lhs = f["r_portfolio_gross"] - f["r_policy"]
    rhs = f["macro"] + f["equity_weight"] * decision.effects.per_period()[list(EFFECTS)].sum(axis=1)
    assert (lhs - rhs).abs().max() <= TOL
    assert f["identity_gap"].abs().max() <= TOL


def test_scaled_effects_are_raw_times_equity_weight(decision: DecisionAttribution) -> None:
    s = decision.scaled
    for col in EFFECTS:
        np.testing.assert_allclose(s[f"{col}_scaled"], s[col] * s["equity_weight"], atol=1e-18)
    assert (s["equity_weight"] < 1).any()  # the crisis months must actually scale something


def test_joint_identity_holds_with_cash_in_policy(sample: SampleData) -> None:
    """With w_pol < 1 the identity needs the policy benchmark, not the raw benchmark."""
    nw = dict(sample.policy.neutral_weights)
    biggest = max((k for k in nw if k != CASH_KEY), key=lambda k: nw[k])
    nw[biggest] = round(nw[biggest] - 0.10, 6)
    nw[CASH_KEY] = 0.10
    bands = {k: (max(0.0, v - 0.05), min(1.0, v + 0.05)) for k, v in nw.items()}
    bands[CASH_KEY] = (0.0, 0.40)
    policy = dataclasses.replace(sample.policy, neutral_weights=nw, bands=bands)
    r = _run(sample, policy=policy)
    assert r.per_period["policy_weight"].iloc[0] == pytest.approx(0.9)
    assert r.per_period["identity_gap"].abs().max() <= TOL
    assert r.linked.linked_sum == pytest.approx(r.linked.geometric_active, abs=1e-9)


# --- linking -----------------------------------------------------------------


@pytest.mark.parametrize("method", ["carino", "menchero"])
def test_linked_effects_sum_to_geometric_active(sample: SampleData, method: str) -> None:
    r = _run(sample, method=method)
    assert r.linked.method == method and r.method == method
    f = r.per_period
    geo = float((1 + f["r_portfolio_gross"]).prod() - (1 + f["r_policy"]).prod())
    assert r.linked.geometric_active == pytest.approx(geo, abs=1e-12)
    assert abs(r.linked.linked_sum - geo) <= 1e-9
    assert set(r.linked.totals) == {"macro", *EFFECTS}


def test_carino_and_menchero_differ_but_both_close(sample: SampleData) -> None:
    c, m = _run(sample, method="carino").linked, _run(sample, method="menchero").linked
    assert not np.allclose(c.factors.to_numpy(), m.factors.to_numpy())
    assert c.linked_sum == pytest.approx(m.linked_sum, abs=1e-9)


def test_carino_limit_when_returns_equal() -> None:
    idx = pd.period_range("2020-01", periods=3, freq="M")
    rp = pd.Series([0.02, 0.01, -0.01], index=idx)
    f = linking_factors(rp, rp.copy(), "carino")
    big = float((1 + rp).prod() - 1)
    np.testing.assert_allclose(f.to_numpy(), (1 / (1 + rp.to_numpy())) / (1 / (1 + big)))


def test_linking_refuses_incomplete_effects(decision: DecisionAttribution) -> None:
    f = decision.per_period
    with pytest.raises(BrinsonError, match="include every component"):
        link_multi_period(f[list(EFFECTS)], portfolio_returns=f["r_portfolio_gross"], benchmark_returns=f["r_policy"])


def test_linking_uses_scaled_not_raw_effects(decision: DecisionAttribution) -> None:
    raw_alloc = float((decision.effects.per_period()["allocation"] * decision.linked.factors).sum())
    assert decision.linked.totals["allocation"] != pytest.approx(raw_alloc, abs=1e-6)


# --- benchmark-identical portfolio, variants ---------------------------------


def test_portfolio_identical_to_benchmark_gives_zero_effects(sample: SampleData) -> None:
    b = sample.benchmark
    holdings = b.assign(security_id="ID_" + b["sector"].str.replace(r"\W", "", regex=True))[
        ["date", "security_id", "weight", "sector"]]
    prices = b.assign(security_id=holdings["security_id"])[["date", "security_id", "total_return"]]
    for variant in ("fachler", "hood_beebower"):
        e = single_period_effects(holdings, b, prices, variant)  # type: ignore[arg-type]
        assert e.table[list(EFFECTS)].abs().to_numpy().max() <= 1e-15


def test_fachler_and_hood_beebower_differ_in_allocation_not_total(sample: SampleData) -> None:
    f = single_period_effects(sample.holdings, sample.benchmark, sample.prices, "fachler").table
    hb = single_period_effects(sample.holdings, sample.benchmark, sample.prices, "hood_beebower").table
    assert (f["allocation"] - hb["allocation"]).abs().max() > 1e-4  # sector split differs
    np.testing.assert_allclose(f.groupby("period")["allocation"].sum(), hb.groupby("period")["allocation"].sum(), atol=TOL)
    np.testing.assert_allclose(f["selection_effect"], hb["selection_effect"], atol=0)
    np.testing.assert_allclose(f["interaction"], hb["interaction"], atol=0)
    np.testing.assert_allclose(f.groupby("period")["total"].sum(), hb.groupby("period")["total"].sum(), atol=TOL)


def test_hood_beebower_rewards_overweights_in_rising_market() -> None:
    """The artefact Fachler removes: an overweight in a sector that rose LESS than the benchmark."""
    h, b, p = _toy(
        ["2025-01-31"],
        {"Auto": [(0.5, 0.02)], "Bankex": [(0.5, 0.10)]},
        {"Auto": [("A", 0.8, 0.02)], "Bankex": [("B", 0.2, 0.10)]},
    )
    fa = single_period_effects(h, b, p, "fachler").table.set_index("sector")
    hb = single_period_effects(h, b, p, "hood_beebower").table.set_index("sector")
    assert hb.loc["Auto", "allocation"] > 0  # looks good just because Auto went up
    assert fa.loc["Auto", "allocation"] < 0  # correctly negative: Auto lagged the benchmark


# --- real-world cases --------------------------------------------------------


def test_sector_held_but_not_in_benchmark() -> None:
    h, b, p = _toy(
        ["2025-01-31", "2025-02-28"],
        {"Auto": [(0.6, 0.03), (0.6, 0.01)], "Bankex": [(0.4, 0.01), (0.4, 0.02)]},
        {"Auto": [("A", 0.5, 0.04), ("A", 0.5, 0.00)], "Bankex": [("B", 0.3, 0.02), ("B", 0.3, 0.03)],
         "Realty": [("R", 0.2, 0.08), ("R", 0.2, -0.05)]},
    )
    e = single_period_effects(h, b, p)
    realty = e.table[e.table["sector"] == "Realty"]
    assert (realty["w_b"] == 0).all() and (realty["w_p"] > 0).all()
    assert (realty["selection_effect"] == 0).all() and (realty["interaction"] == 0).all()
    assert (realty["allocation"] != 0).all()
    assert sum(o.kind == "held_not_in_benchmark" and o.sector == "Realty" for o in e.orphans) == 2
    pp = e.per_period()
    assert (pp[list(EFFECTS)].sum(axis=1) - pp["active_equity_book"]).abs().max() <= TOL


def test_sector_in_benchmark_but_not_held(decision: DecisionAttribution) -> None:
    orphans = [o for o in decision.effects.orphans if o.kind == "benchmark_not_held"]
    t = decision.effects.table
    out = t[t["in_benchmark"] & ~t["held"]]
    assert len(orphans) == len(out)
    if out.empty:
        pytest.skip("sample has no benchmark-only sector")
    assert (out["selection_effect"] == 0).all() and (out["interaction"] == 0).all()
    np.testing.assert_allclose(out["allocation"], -out["w_b"] * (out["r_b"] - out["r_b_total"]), atol=0)


def test_benchmark_only_sector_toy() -> None:
    h, b, p = _toy(
        ["2025-01-31"],
        {"Auto": [(0.5, 0.05)], "Metal": [(0.5, -0.04)]},
        {"Auto": [("A", 1.0, 0.06)]},
    )
    e = single_period_effects(h, b, p).table.set_index("sector")
    assert e.loc["Metal", "allocation"] == pytest.approx(-0.5 * (-0.04 - 0.005))  # being out of a loser helps
    assert e.loc["Metal", "selection_effect"] == 0 and e.loc["Metal", "interaction"] == 0


def test_sector_with_zero_weight_in_both_is_exactly_zero() -> None:
    h, b, p = _toy(
        ["2025-01-31", "2025-02-28"],
        {"Auto": [(0.5, 0.02), (0.5, 0.01)], "Bankex": [(0.5, 0.01), (0.5, 0.03)], "Power": [(0.0, 0.07), (0.0, 0.0)]},
        {"Auto": [("A", 0.5, 0.03), ("A", 0.6, 0.00)], "Bankex": [("B", 0.5, 0.00), ("B", 0.4, 0.02)]},
    )
    t = single_period_effects(h, b, p).table
    for sector in ("Power", "Energy"):  # Power: zero-weight benchmark row; Energy: no row at all
        rows = t[t["sector"] == sector]
        assert len(rows) == 2
        assert (rows[list(EFFECTS)] == 0.0).all().all()
    assert set(t["sector"]) == set(SECTORS)


def test_missing_price_is_dropped_and_recorded(sample: SampleData) -> None:
    h = sample.holdings
    victim = h.iloc[5]
    prices = sample.prices[~((sample.prices["date"] == victim["date"]) & (sample.prices["security_id"] == victim["security_id"]))]
    r = _run(sample, prices=prices)
    dropped = r.effects.dropped
    assert len(dropped) == 1
    d = dropped[0]
    assert (d.security_id, d.date, d.weight) == (victim["security_id"], victim["date"].date(), pytest.approx(victim["weight"]))
    assert "no price" in d.reason
    period = victim["date"].to_period("M")
    t = r.effects.table
    assert t.loc[t["period"] == period, "w_p"].sum() == pytest.approx(1.0)  # remaining names rescaled
    assert r.per_period["identity_gap"].abs().max() <= TOL
    assert "dropped holdings (no price): 1" in r.report()


# --- summary, intervals, terminology -----------------------------------------


def test_by_sector_totals_reconcile(decision: DecisionAttribution) -> None:
    s = decision.by_sector
    body, total = s.drop(index="TOTAL"), s.loc["TOTAL"]
    assert set(body.index) == set(SECTORS)
    for col in EFFECTS:
        assert body[col].sum() == pytest.approx(total[col], abs=1e-12)
        assert total[col] == pytest.approx(decision.linked.totals[col], abs=1e-12)
    assert total["w_p"] == pytest.approx(1.0) and total["w_b"] == pytest.approx(1.0)
    assert total[list(EFFECTS)].sum() + decision.linked.totals["macro"] == pytest.approx(decision.linked.geometric_active, abs=1e-9)


def test_confidence_intervals_for_each_effect(decision: DecisionAttribution) -> None:
    for col in EFFECTS:
        ci = decision.intervals[col]
        assert ci.lower <= ci.point <= ci.upper
        assert ci.n_iterations == 2000 and ci.block_length == 6
        assert ci.point == pytest.approx(decision.per_period[col].mean() * 12)


def test_residual_vs_reported_is_separate(decision: DecisionAttribution, sample: SampleData) -> None:
    res = decision.residual_vs_reported
    assert res is not None
    fee = 1.02 ** (1 / 12) - 1
    np.testing.assert_allclose(res.to_numpy(), -fee, atol=1e-12)  # sample: exactly the management fee


def test_terminology_selection_effect_is_never_a_residual() -> None:
    """'selection_effect' is the measured stock-picking effect, never a leftover."""
    pkg = pathlib.Path(__file__).resolve().parents[1] / "attribution"
    offenders = []
    for path in pkg.glob("*.py"):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            code = line.split("#", 1)[0]
            if re.search(r"selection_effect\w*\s*=.*(residual|specific)", code) or re.search(
                r"(residual|specific)\w*\s*=.*selection_effect", code
            ):
                offenders.append(f"{path.name}:{n}")
    assert offenders == []
    assert "selection_effect" in EFFECTS and "selection" not in EFFECTS


def test_report_mentions_method_and_reconciles(decision: DecisionAttribution) -> None:
    text = decision.report()
    assert "linking carino" in text and "variant fachler" in text
    assert math.isclose(decision.linked.linked_sum, decision.linked.geometric_active, abs_tol=1e-9)


def test_inputs_must_cover_same_months(sample: SampleData) -> None:
    b = sample.benchmark[sample.benchmark["date"] != sample.benchmark["date"].max()]
    with pytest.raises(BrinsonError, match="different months"):
        single_period_effects(sample.holdings, b, sample.prices)


def test_scale_requires_every_period(decision: DecisionAttribution) -> None:
    w = decision.per_period["equity_weight"].iloc[1:]
    with pytest.raises(BrinsonError, match="missing"):
        scale_by_exposure(decision.effects, w)
