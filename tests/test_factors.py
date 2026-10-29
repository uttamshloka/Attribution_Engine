"""Properties of the returns-based factor view.

Properties, not numbers, so the tests survive regenerating the sample data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from attribution.brinson import run_decision_attribution
from attribution.factors import (
    CONDITION_LIMIT,
    FACTOR_ORDER,
    MARKET_VS_MACRO_NOTE,
    FactorError,
    FactorMatrix,
    FactorModelResult,
    active_returns,
    build_factor_matrix,
    fit_factor_model,
    inspect_factor_series,
    link_factor_contributions,
    orthogonalise,
    rolling_exposures,
)
from attribution.macro_layer import benchmark_total_return
from attribution.result import FactorView
from attribution.sample import SampleData
from attribution.schemas import FACTORS


@pytest.fixture(scope="module")
def matrix(sample: SampleData) -> FactorMatrix:
    return build_factor_matrix(sample.factor_returns, sample.benchmark, sample.cash_returns)


@pytest.fixture(scope="module")
def active(sample: SampleData) -> pd.Series:
    return active_returns(sample.portfolio_returns, sample.benchmark)


@pytest.fixture(scope="module")
def model(active: pd.Series, matrix: FactorMatrix) -> FactorModelResult:
    return fit_factor_model(active, matrix)  # with bootstrap


def _fit(active: pd.Series, matrix: FactorMatrix | pd.DataFrame, **kw: object) -> FactorModelResult:
    return fit_factor_model(active, matrix, bootstrap=False, **kw)  # type: ignore[arg-type]


def _factor_frame(sample: SampleData, **replace: pd.Series) -> pd.DataFrame:
    """sample factor_returns with some series replaced (values aligned to the sample dates)."""
    wide = sample.factor_returns.pivot(index="date", columns="factor", values="total_return")
    for name, values in replace.items():
        wide[name] = np.asarray(values, dtype=float)
    return wide.reset_index().melt(id_vars="date", var_name="factor", value_name="total_return")


# --- the headline identity ---------------------------------------------------


def test_identity_closes(model: FactorModelResult, active: pd.Series) -> None:
    lhs = active.mean()
    rhs = model.alpha_monthly + sum(r.exposure * r.factor_mean_annual / 12 for r in model.rows)
    assert abs(lhs - rhs) <= 1e-10
    assert abs(model.total - active.mean() * 12) <= 1e-10
    assert abs(model.identity_gap) <= 1e-10
    assert model.specific_return == pytest.approx(model.alpha_monthly * 12, abs=1e-15)


def test_per_period_decomposition_sums_to_active(model: FactorModelResult, active: pd.Series) -> None:
    per = model.per_period
    np.testing.assert_allclose(per[[*model.order, "specific"]].sum(axis=1), active.to_numpy(), atol=1e-14)


# --- market factor handling --------------------------------------------------


def test_market_regressor_is_benchmark_minus_cash(sample: SampleData, matrix: FactorMatrix) -> None:
    rb = benchmark_total_return(sample.benchmark).to_numpy()
    rc = sample.cash_returns["total_return"].to_numpy()
    np.testing.assert_allclose(matrix.data["market"].to_numpy(), rb - rc, atol=0)
    assert "active beta" in matrix.transformations["market"]


def test_inspection_classifies_sample_series(sample: SampleData) -> None:
    ins = inspect_factor_series(sample.factor_returns, sample.benchmark, sample.cash_returns, verbose=False)
    assert ins.row("market").classification in ("index_total_return", "benchmark_itself", "excess_market_return")
    for f in FACTORS[1:]:
        assert ins.row(f).classification == "tilt_return", f
    assert "FACTOR SERIES INSPECTION" in ins.report()


def test_inspection_detects_each_kind_of_market_series(sample: SampleData) -> None:
    rb = benchmark_total_return(sample.benchmark).to_numpy()
    rc = sample.cash_returns["total_return"].to_numpy()
    cases = {"benchmark_itself": rb, "excess_market_return": rb - rc}
    for expected, series in cases.items():
        ins = inspect_factor_series(_factor_frame(sample, market=series), sample.benchmark, sample.cash_returns, verbose=False)
        assert ins.row("market").classification == expected


def test_supplied_market_source_converts_total_to_excess(sample: SampleData) -> None:
    rb = benchmark_total_return(sample.benchmark).to_numpy()
    rc = sample.cash_returns["total_return"].to_numpy()
    fm = build_factor_matrix(_factor_frame(sample, market=rb), sample.benchmark, sample.cash_returns, market_source="supplied")
    np.testing.assert_allclose(fm.data["market"].to_numpy(), rb - rc, atol=1e-15)
    fm2 = build_factor_matrix(_factor_frame(sample, market=rb - rc), sample.benchmark, sample.cash_returns, market_source="supplied")
    np.testing.assert_allclose(fm2.data["market"].to_numpy(), rb - rc, atol=1e-15)
    assert "used as is" in fm2.transformations["market"]


def test_index_like_style_series_converted_to_tilt(sample: SampleData) -> None:
    rb = benchmark_total_return(sample.benchmark).to_numpy()
    index_like = rb * 1.05 + 0.001
    fm = build_factor_matrix(_factor_frame(sample, value=index_like), sample.benchmark, sample.cash_returns)
    np.testing.assert_allclose(fm.data["value"].to_numpy(), index_like - rb, atol=1e-15)
    assert "subtracting the benchmark" in fm.transformations["value"]


# --- orthogonalisation -------------------------------------------------------


def test_orthogonalised_columns_are_orthogonal_and_first_unchanged(matrix: FactorMatrix) -> None:
    o = orthogonalise(matrix.data)
    Q = o.data.to_numpy()
    gram = Q.T @ Q
    off = gram - np.diag(np.diag(gram))
    assert np.abs(off).max() <= 1e-12 * np.abs(np.diag(gram)).max()
    np.testing.assert_allclose(o.data["market"], matrix.data["market"], atol=0)
    assert o.order == FACTOR_ORDER
    assert "incremental" in o.note


def test_orthogonalisation_keeps_total_but_redistributes(active: pd.Series, matrix: FactorMatrix) -> None:
    orth = _fit(active, matrix, orthogonalise_factors=True)
    raw = _fit(active, matrix, orthogonalise_factors=False)
    assert orth.total == pytest.approx(raw.total, abs=1e-10)
    assert orth.specific_return == pytest.approx(raw.specific_return, abs=1e-12)
    assert sum(r.contribution for r in orth.rows) == pytest.approx(sum(r.contribution for r in raw.rows), abs=1e-10)
    diffs = [abs(a.contribution - b.contribution) for a, b in zip(orth.rows, raw.rows, strict=True)]
    assert max(diffs) > 1e-5  # individual lines do move


def test_reversing_order_changes_lines_not_total(active: pd.Series, matrix: FactorMatrix) -> None:
    fwd = _fit(active, matrix)
    rev = _fit(active, matrix, order=tuple(reversed(FACTOR_ORDER)))
    assert rev.order == tuple(reversed(FACTOR_ORDER))
    assert fwd.total == pytest.approx(rev.total, abs=1e-10)
    assert fwd.specific_return == pytest.approx(rev.specific_return, abs=1e-12)
    changed = [abs(fwd.row(f).contribution - rev.row(f).contribution) for f in FACTORS]
    assert max(changed) > 1e-5


def test_perfect_collinearity_rejected(matrix: FactorMatrix) -> None:
    bad = matrix.data.copy()
    bad["value"] = 2 * bad["size"] - bad["market"]
    with pytest.raises(FactorError, match="linear combination"):
        orthogonalise(bad)


def test_high_condition_number_adds_limitation(active: pd.Series, matrix: FactorMatrix) -> None:
    rng = np.random.default_rng(1)
    coll = matrix.data.copy()
    coll["value"] = coll["size"] + rng.normal(0, 0.001, len(coll))
    r = _fit(active, coll)
    assert r.condition_number_raw > CONDITION_LIMIT
    assert any("Condition number" in x for x in r.limitations)
    assert r.vif_raw["value"] > 10 and r.vif_used["value"] < r.vif_raw["value"]
    assert r.total == pytest.approx(active.mean() * 12, abs=1e-10)


def test_sample_condition_number_reported(model: FactorModelResult) -> None:
    assert model.condition_number_raw >= 1.0
    assert set(model.vif_raw) == set(FACTORS) == set(model.vif_used)
    if model.condition_number_raw <= CONDITION_LIMIT:
        assert not any("Condition number" in x for x in model.limitations)
    assert any("incremental" in x for x in model.limitations)


def test_diagnostics_in_range(model: FactorModelResult) -> None:
    assert 0.0 <= model.r_squared <= 1.0
    assert model.adj_r_squared <= model.r_squared
    assert 0.0 <= model.durbin_watson <= 4.0
    assert model.tracking_error > 0 and model.residual_volatility > 0


# --- the cross-framework check -----------------------------------------------


def test_size_tilt_recovered_positive_and_significant(model: FactorModelResult) -> None:
    """The sample was built with a deliberate small/mid-cap tilt. The model MUST find it."""
    size = model.row("size")
    msg = (f"MODEL FAILED TO RECOVER THE SIZE TILT: exposure {size.exposure:+.3f}, t {size.t_stat:+.2f}, "
           f"CI {size.exposure_ci}. The sample is built to load positively on size; check the factor model.")
    assert size.exposure > 0, msg
    assert size.exposure_raw > 0, msg
    assert size.t_stat > 2.0, msg
    assert size.exposure_ci is not None and size.exposure_ci.lower > 0, msg
    assert size.exposure == max(r.exposure for r in model.rows), msg


def test_size_exposure_close_to_ground_truth(sample: SampleData, model: FactorModelResult) -> None:
    """The recovered exposure should sit near the true active size loading built into the sample."""
    chars = sample.characteristics.set_index("security_id")
    load = chars["loading_size"]
    h = sample.holdings
    port = (h["weight"] * h["security_id"].map(load)).groupby(h["date"]).sum()
    prices = sample.prices.pivot(index="date", columns="security_id", values="total_return")
    caps = prices.shift(1).fillna(0.0).add(1.0).cumprod().mul(chars.loc[prices.columns, "initial_market_cap_cr"])
    bench = caps.div(caps.sum(axis=1), axis=0).mul(load.loc[prices.columns]).sum(axis=1)
    true_active = float((port - bench).mean())
    assert true_active > 0
    assert model.row("size").exposure == pytest.approx(true_active, rel=0.35)


# --- benchmark-identical portfolio -------------------------------------------


def test_portfolio_identical_to_benchmark_is_all_zero(sample: SampleData, matrix: FactorMatrix) -> None:
    rb = benchmark_total_return(sample.benchmark)
    same = pd.DataFrame({"date": rb.index, "total_return": rb.to_numpy()})
    act = active_returns(same, sample.benchmark)
    assert (act == 0).all()
    r = fit_factor_model(act, matrix)
    assert all(row.contribution == 0.0 and row.exposure == 0.0 for row in r.rows)
    assert r.specific_return == 0.0 and r.total == 0.0


# --- bootstrap, result object, rolling, linking --------------------------------


def test_every_contribution_and_specific_has_an_interval(model: FactorModelResult) -> None:
    for r in model.rows:
        assert r.ci is not None and r.exposure_ci is not None
        assert r.ci.point == pytest.approx(r.contribution, abs=1e-12)
        assert r.exposure_ci.point == pytest.approx(r.exposure, abs=1e-12)
        assert r.ci.n_iterations == 2000 and r.ci.block_length == 6
    assert model.specific_ci is not None
    assert model.specific_ci.point == pytest.approx(model.specific_return, abs=1e-12)
    text = model.report()
    assert "spans zero" in text and "RECONCILIATION" in text


def test_to_factor_view_matches_contract(model: FactorModelResult) -> None:
    fv = model.to_factor_view()
    assert isinstance(fv, FactorView)
    assert tuple(f.factor for f in fv.factors) == FACTORS
    assert fv.total == pytest.approx(model.active_annualised, abs=1e-10)
    assert fv.diagnostics.n_obs == model.n_obs
    assert "Gram-Schmidt" in fv.diagnostics.method


def test_market_vs_macro_note_present(model: FactorModelResult) -> None:
    assert model.note == MARKET_VS_MACRO_NOTE
    assert "TOTALS" in model.note and "do not have to" in model.note
    assert "NOTE" in model.report()


def test_rolling_exposures_are_separate_from_totals(active: pd.Series, matrix: FactorMatrix) -> None:
    before = _fit(active, matrix)
    path = rolling_exposures(active, matrix, window=36)
    after = _fit(active, matrix)
    assert len(path) == len(active) - 36 + 1
    assert list(path.columns) == ["alpha_monthly", *FACTOR_ORDER]
    assert after.total == before.total and after.rows == before.rows  # untouched by the rolling fit
    assert not np.isclose(path["size"].iloc[0], before.row("size").exposure)
    with pytest.raises(FactorError):
        rolling_exposures(active, matrix, window=6)


def test_linked_factor_contributions_sum_to_geometric_net_active(sample: SampleData, model: FactorModelResult) -> None:
    L = link_factor_contributions(model, sample.portfolio_returns, sample.benchmark)
    rp = sample.portfolio_returns["total_return"].to_numpy()
    rb = benchmark_total_return(sample.benchmark).to_numpy()
    geo = float(np.prod(1 + rp) - np.prod(1 + rb))
    assert abs(L.linked_sum - geo) <= 1e-9


def test_totals_of_both_views_reconcile(sample: SampleData, model: FactorModelResult) -> None:
    """Decision view (gross vs policy) + fee drag = factor view (net vs benchmark), geometrically."""
    dec = run_decision_attribution(sample.holdings, sample.benchmark, sample.prices, sample.regime, sample.policy,
                                   sample.cash_returns, sample.portfolio_returns, bootstrap=False)
    fac = link_factor_contributions(model, sample.portfolio_returns, sample.benchmark)
    pp = dec.per_period
    fee_effect = float((1 + pp["r_portfolio_reported"]).prod() - (1 + pp["r_portfolio_gross"]).prod())
    assert sample.policy.neutral_equity_exposure == 1.0  # policy benchmark == benchmark here
    assert dec.linked.linked_sum + fee_effect == pytest.approx(fac.linked_sum, abs=1e-9)
