"""Properties of the macro (asset-class exposure) layer.

These assert properties, not specific numbers, so they survive regenerating
the sample data.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from attribution.macro_layer import (
    CASH_KEY,
    EQUITY_KEY,
    INTEGER_STATE_CODES,
    MacroLayerError,
    MacroResult,
    asset_class_allocation,
    benchmark_total_return,
    compute_macro_contribution,
    max_drawdown,
)
from attribution.sample import SampleData
from attribution.schemas import Policy


def _run(sample: SampleData, **overrides: object) -> MacroResult:
    args: dict[str, object] = {
        "regime": sample.regime,
        "policy": sample.policy,
        "equity_returns": benchmark_total_return(sample.benchmark),
        "cash_returns": sample.cash_returns,
        "bootstrap": False,
    }
    args.update(overrides)
    return compute_macro_contribution(**args)  # type: ignore[arg-type]


@pytest.fixture(scope="module")
def macro(sample: SampleData) -> MacroResult:
    return compute_macro_contribution(
        sample.regime, sample.policy, benchmark_total_return(sample.benchmark), sample.cash_returns
    )


def _with_cash_band(policy: Policy, band: tuple[float, float]) -> Policy:
    bands = dict(policy.bands)
    bands[CASH_KEY] = band
    return dataclasses.replace(policy, bands=bands)


# --- exactly zero when the overlay never fires -------------------------------


def test_window_where_overlay_never_fires_is_exactly_zero(sample: SampleData) -> None:
    """Months before the crisis are at policy every day: contribution must be exactly 0.0."""
    cutoff = pd.Timestamp("2024-04-30")
    bench = benchmark_total_return(sample.benchmark)
    result = _run(
        sample,
        regime=sample.regime[sample.regime["date"] <= cutoff],
        equity_returns=bench[bench.index <= cutoff],
        cash_returns=sample.cash_returns[sample.cash_returns["date"] <= cutoff],
    )
    assert (result.contribution == 0.0).all()
    assert result.total == 0.0
    assert not np.signbit(result.total)  # not even -0.0
    assert result.risk.share_periods_at_policy == 1.0
    assert result.risk.n_episodes == 0


def test_at_policy_months_are_exactly_zero_in_full_sample(macro: MacroResult) -> None:
    at_policy = macro.per_period["phase"] == "at_policy"
    assert at_policy.any()
    assert (macro.per_period.loc[at_policy, "contribution"] == 0.0).all()


# --- multi-asset Brinson-Fachler vs the two-asset shorthand -----------------


def test_multi_asset_sum_equals_two_asset_shorthand(macro: MacroResult) -> None:
    p = macro.per_period
    shorthand = (p["equity_weight"] - p["policy_weight"]) * (p["equity_return"] - p["cash_return"])
    np.testing.assert_allclose(p["contribution"], shorthand, atol=1e-15)


def test_multi_asset_identity_on_random_inputs() -> None:
    rng = np.random.default_rng(0)
    idx = pd.period_range("2020-01", periods=24, freq="M")
    we = rng.uniform(0.4, 1.0, 24)
    wp = 0.8
    re_, rc = rng.normal(0.01, 0.05, 24), rng.normal(0.005, 0.001, 24)
    full = asset_class_allocation(
        pd.DataFrame({EQUITY_KEY: we, CASH_KEY: 1 - we}, index=idx),
        {EQUITY_KEY: wp, CASH_KEY: 1 - wp},
        pd.DataFrame({EQUITY_KEY: re_, CASH_KEY: rc}, index=idx),
    )
    np.testing.assert_allclose(full.to_numpy(), (we - wp) * (re_ - rc), atol=1e-15)


def test_equity_line_alone_drops_the_effect_and_is_refused(macro: MacroResult) -> None:
    """The classic mistake: summing Brinson-Fachler over the equity line only.

    With a 100%-equity policy, r_policy equals the equity return, so the
    equity-line term is identically zero and the whole effect sits in the
    cash (funding) leg. The equity-only sum must differ from the true
    contribution, and asset_class_allocation must refuse to compute it.
    """
    p = macro.per_period
    r_policy = p["policy_weight"] * p["equity_return"] + (1 - p["policy_weight"]) * p["cash_return"]
    equity_line_only = ((p["equity_weight"] - p["policy_weight"]) * (p["equity_return"] - r_policy)).sum()
    assert abs(macro.total) > 1e-6
    assert abs(equity_line_only - macro.total) > 0.5 * abs(macro.total)

    idx = p.index
    with pytest.raises(MacroLayerError, match="funding leg"):
        asset_class_allocation(
            pd.DataFrame({EQUITY_KEY: p["equity_weight"]}, index=idx),
            {EQUITY_KEY: 1.0},
            pd.DataFrame({EQUITY_KEY: p["equity_return"]}, index=idx),
        )
    with pytest.raises(MacroLayerError, match="missing asset class"):
        asset_class_allocation(
            pd.DataFrame({EQUITY_KEY: p["equity_weight"]}, index=idx),
            {EQUITY_KEY: 1.0, CASH_KEY: 0.0},
            pd.DataFrame({EQUITY_KEY: p["equity_return"], CASH_KEY: p["cash_return"]}, index=idx),
        )


def test_overlay_minus_policy_book_equals_contribution(macro: MacroResult) -> None:
    p = macro.per_period
    overlay = p["equity_weight"] * p["equity_return"] + (1 - p["equity_weight"]) * p["cash_return"]
    policy = p["policy_weight"] * p["equity_return"] + (1 - p["policy_weight"]) * p["cash_return"]
    np.testing.assert_allclose(overlay - policy, p["contribution"], atol=1e-15)


# --- band clipping -----------------------------------------------------------


def test_band_clipping_is_recorded(sample: SampleData) -> None:
    tight = _with_cash_band(sample.policy, (0.0, 0.2))  # equity may not go below 0.8
    result = _run(sample, policy=tight)
    below = int((sample.regime["equity_exposure"] < 0.8).sum())
    assert below > 0
    assert len(result.clip_events) == below
    assert all(e.bound == "lower" and e.applied == pytest.approx(0.8) and e.requested < 0.8 for e in result.clip_events)
    assert result.per_period["equity_weight"].min() >= 0.8 - 1e-12
    assert "clipped" in result.report()


def test_no_clipping_recorded_when_inside_band(macro: MacroResult) -> None:
    lo, hi = macro.equity_band
    assert macro.per_period["equity_weight"].between(lo - 1e-12, hi + 1e-12).all()
    assert macro.clip_events == ()


def test_upper_clipping_recorded() -> None:
    days = pd.bdate_range("2025-01-01", "2025-02-28")
    regime = pd.DataFrame({"date": days, "raw_state": "normal", "confirmed_state": "normal", "equity_exposure": 1.0})
    from attribution.sample import generate_sample

    base = generate_sample().policy
    nw = dict(base.neutral_weights)
    # Move 10% of the largest sector into cash: policy equity 0.9, band 0.8-0.95.
    biggest = max((k for k in nw if k != CASH_KEY), key=lambda k: nw[k])
    nw[biggest] = round(nw[biggest] - 0.10, 6)
    nw[CASH_KEY] = 0.10
    bands = {k: (max(0.0, v - 0.05), min(1.0, v + 0.05)) for k, v in nw.items()}
    bands[CASH_KEY] = (0.05, 0.20)
    policy = dataclasses.replace(base, neutral_weights=nw, bands=bands)
    months = pd.to_datetime(["2025-01-31", "2025-02-28"])
    result = compute_macro_contribution(
        regime, policy,
        pd.DataFrame({"date": months, "total_return": [0.02, 0.01]}),
        pd.DataFrame({"date": months, "total_return": [0.005, 0.005]}),
        bootstrap=False,
    )
    assert len(result.clip_events) == len(days)
    assert all(e.bound == "upper" and e.applied == pytest.approx(0.95) for e in result.clip_events)


# --- two-phase split ---------------------------------------------------------


def test_phases_sum_to_net(macro: MacroResult) -> None:
    for e in macro.episodes:
        assert e.earned_falling + e.given_back_recovering == pytest.approx(e.net, abs=1e-15)
    ph = macro.phases
    assert ph.earned_falling + ph.given_back_recovering == pytest.approx(ph.net_episodes, abs=1e-15)
    assert ph.net_episodes + ph.off_policy_outside_episodes == pytest.approx(macro.total, abs=1e-15)


def test_trough_is_benchmark_minimum_within_episode(macro: MacroResult) -> None:
    p = macro.per_period
    for e in macro.episodes:
        seg = p.loc[e.start:e.end]
        path = (1 + seg["benchmark_return"]).cumprod()
        assert e.trough == str(path.idxmin())
        assert (seg.loc[: e.trough, "phase"] == "earned_falling").all()
        assert (seg.loc[e.trough:, "phase"].iloc[1:] == "given_back_recovering").all()


def test_sample_crisis_shows_earn_then_give_back(macro: MacroResult) -> None:
    """The sample crisis has one sharp fall then a recovery: the split must be visible."""
    assert macro.risk.n_episodes >= 1
    crisis = max(macro.episodes, key=lambda e: e.n_derisked_days)
    assert crisis.earned_falling > 0
    assert crisis.given_back_recovering < 0


# --- risk metrics ------------------------------------------------------------


def test_capture_ratios_within_sensible_bounds(macro: MacroResult) -> None:
    r = macro.risk
    assert r.upside_capture is not None and r.downside_capture is not None
    assert 0.0 < r.upside_capture <= 1.05
    assert 0.0 < r.downside_capture <= 1.05
    assert r.downside_capture < 1.0  # de-risking into falls must reduce downside capture
    assert r.policy_upside_capture == pytest.approx(1.0)  # policy book here is 100% benchmark equity
    assert r.policy_downside_capture == pytest.approx(1.0)


def test_drawdown_metrics_consistent(macro: MacroResult) -> None:
    r = macro.risk
    for d in (r.overlay_drawdown, r.policy_drawdown):
        assert d.max_drawdown <= 0.0
        assert d.duration_periods >= 0
        assert d.recovery_periods is None or d.recovery_periods >= 1
    assert r.drawdown_difference == pytest.approx(r.overlay_drawdown.max_drawdown - r.policy_drawdown.max_drawdown)


def test_drawdown_helper_known_path() -> None:
    idx = pd.period_range("2020-01", periods=5, freq="M")
    d = max_drawdown(pd.Series([0.10, -0.20, -0.10, 0.20, 0.30], index=idx))
    assert d.max_drawdown == pytest.approx(0.792 / 1.1 - 1.0)  # peak 1.10, trough 0.792
    assert (d.peak, d.trough, d.duration_periods) == ("2020-01", "2020-03", 2)
    assert d.recovered_at == "2020-05" and d.recovery_periods == 2


def test_share_at_policy_and_episode_lengths(macro: MacroResult) -> None:
    r = macro.risk
    assert 0.0 <= r.share_periods_at_policy < 1.0
    assert r.n_episodes == len(r.episode_lengths_periods) == len(r.episode_lengths_days)
    assert all(n >= 1 for n in r.episode_lengths_periods)
    assert sum(r.episode_lengths_days) == int((macro.per_period["derisked_days"]).sum())


# --- derisk_mode -------------------------------------------------------------


def test_within_equity_mode_returns_exact_zero_and_says_so(sample: SampleData) -> None:
    policy = dataclasses.replace(sample.policy, derisk_mode="within_equity")
    result = _run(sample, policy=policy)
    assert result.applicable is False
    assert (result.contribution == 0.0).all() and result.total == 0.0
    assert "within_equity" in result.message and "NOT APPLICABLE" in result.message
    assert "NOT APPLICABLE" in result.report()
    assert result.ci is None


# --- regime state coding -----------------------------------------------------


def test_string_and_integer_states_give_identical_results(sample: SampleData, macro: MacroResult) -> None:
    to_int = {v: k for k, v in INTEGER_STATE_CODES.items()}
    coded = sample.regime.assign(
        raw_state=sample.regime["raw_state"].map(to_int).astype(int),
        confirmed_state=sample.regime["confirmed_state"].map(to_int).astype(int),
    )
    assert coded["confirmed_state"].dtype.kind == "i"
    result = _run(sample, regime=coded)
    pd.testing.assert_series_equal(result.contribution, macro.contribution)
    assert result.episodes[0].states == macro.episodes[0].states


def test_unknown_integer_state_rejected(sample: SampleData) -> None:
    bad = sample.regime.assign(confirmed_state=7)
    with pytest.raises(MacroLayerError, match="unknown integer state"):
        _run(sample, regime=bad)


def test_unknown_string_state_rejected(sample: SampleData) -> None:
    from attribution.schemas import SchemaValidationError

    bad = sample.regime.copy()
    bad.loc[bad.index[0], "confirmed_state"] = "panic"
    with pytest.raises(SchemaValidationError):
        _run(sample, regime=bad)


# --- alignment and inputs ----------------------------------------------------


def test_monthly_weight_is_time_average_of_daily(sample: SampleData, macro: MacroResult) -> None:
    reg = sample.regime
    expected = reg.groupby(reg["date"].dt.to_period("M"))["equity_exposure"].mean()
    got = macro.per_period["equity_weight"]
    np.testing.assert_allclose(got.to_numpy(), expected.reindex(got.index).to_numpy(), atol=1e-12)


def test_missing_regime_month_fails_loudly(sample: SampleData) -> None:
    reg = sample.regime[sample.regime["date"].dt.to_period("M") != pd.Period("2023-03", "M")]
    with pytest.raises(MacroLayerError, match="2023-03"):
        _run(sample, regime=reg)


def test_misaligned_return_months_fail_loudly(sample: SampleData) -> None:
    with pytest.raises(MacroLayerError, match="different months"):
        _run(sample, cash_returns=sample.cash_returns.iloc[1:])


# --- a negative result is reported, never hidden -----------------------------


def test_derisking_through_a_bull_market_is_negative_and_reported() -> None:
    days = pd.bdate_range("2025-01-01", "2025-12-31")
    regime = pd.DataFrame({"date": days, "raw_state": "elevated", "confirmed_state": "elevated", "equity_exposure": 0.7})
    from attribution.sample import generate_sample

    policy = generate_sample().policy
    months = pd.date_range("2025-01-31", periods=12, freq="ME")
    result = compute_macro_contribution(
        regime, policy,
        pd.DataFrame({"date": months, "total_return": 0.02}),
        pd.DataFrame({"date": months, "total_return": 0.005}),
    )
    assert result.total == pytest.approx(12 * (-0.3) * 0.015)
    assert result.total < 0  # not clamped
    text = result.report()
    assert "negative is a valid result" in text
    assert "downside capture" in text and "max drawdown" in text
