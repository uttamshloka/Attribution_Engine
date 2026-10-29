from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attribution import io
from attribution.sample import (
    ACTIVE_CAP,
    ACTIVE_CAP_WIDE,
    BENCHMARK_SECTOR_BANDS,
    BENCHMARK_SECTOR_WEIGHTS,
    MAX_WIDE_SECTORS,
    SampleConfig,
    SampleData,
    generate_sample,
    main,
    tilt_diagnostics,
    write_sample,
)
from attribution.schemas import INDEX_NAMES_NOT_SECTORS, SECTORS, check_holdings_against_master

_TABLES = (
    "security_master",
    "holdings",
    "prices",
    "benchmark",
    "portfolio_returns",
    "factor_returns",
    "cash_returns",
    "regime",
    "characteristics",
    "true_size_exposure",
)


def _bench_returns(sample: SampleData) -> pd.Series:
    b = sample.benchmark
    return (b["weight"] * b["total_return"]).groupby(b["date"]).sum()


def _active_sector_weights(sample: SampleData) -> pd.DataFrame:
    h = sample.holdings.groupby(["date", "sector"])["weight"].sum().unstack(fill_value=0.0)
    bw = sample.benchmark.pivot(index="date", columns="sector", values="weight")
    return h.reindex(columns=bw.columns, fill_value=0.0) - bw


# --- reproducibility and files -----------------------------------------------


def test_reproducible_in_memory() -> None:
    a = generate_sample(SampleConfig(seed=7))
    b = generate_sample(SampleConfig(seed=7))
    for name in _TABLES:
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))
    assert a.policy == b.policy


def test_different_seed_differs() -> None:
    a = generate_sample(SampleConfig(seed=7))
    b = generate_sample(SampleConfig(seed=8))
    assert not a.prices["total_return"].equals(b.prices["total_return"])


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_reproducible_on_disk(tmp_path: Path, fmt: str) -> None:
    first = write_sample(generate_sample(), tmp_path / "a", fmt)
    second = write_sample(generate_sample(), tmp_path / "b", fmt)
    assert first.keys() == second.keys()
    for key in first:
        assert first[key].read_bytes() == second[key].read_bytes(), key


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_every_file_has_synthetic_header_and_loads(tmp_path: Path, fmt: str, sample: SampleData) -> None:
    paths = write_sample(sample, tmp_path, fmt)
    for path in paths.values():
        header = io.read_header_comment(path)
        assert any("SYNTHETIC SAMPLE DATA" in line for line in header), path
    loaded = {
        "security_master": io.load_security_master(paths["security_master"]),
        "holdings": io.load_holdings(paths["holdings"]),
        "prices": io.load_prices(paths["prices"]),
        "benchmark": io.load_benchmark(paths["benchmark"]),
        "portfolio_returns": io.load_portfolio_returns(paths["portfolio_returns"]),
        "factor_returns": io.load_factor_returns(paths["factor_returns"]),
        "cash_returns": io.load_cash_returns(paths["cash_returns"]),
        "regime": io.load_regime_series(paths["regime"]),
    }
    for name, frame in loaded.items():
        pd.testing.assert_frame_equal(frame, getattr(sample, name), check_exact=False, rtol=1e-11, atol=1e-13)
    assert io.load_policy(paths["policy"]) == sample.policy


def test_shape(sample: SampleData) -> None:
    master = sample.security_master
    assert len(master) == 40
    assert set(master["sector"]) == set(SECTORS)
    months = sorted(sample.portfolio_returns["date"].unique())
    assert len(months) == 60
    assert pd.Timestamp(months[-1]) == pd.Timestamp("2026-09-30")
    assert all(pd.Timestamp(m).is_month_end for m in months)
    for frame in (sample.holdings, sample.prices, sample.benchmark, sample.factor_returns, sample.cash_returns):
        assert sorted(frame["date"].unique()) == months
    assert len(sample.prices) == 40 * 60
    assert set(sample.benchmark["sector"]) == set(SECTORS)
    check_holdings_against_master(sample.holdings, master)


def test_end_date_is_configurable() -> None:
    data = generate_sample(SampleConfig(end_date=date(2025, 3, 14), n_months=36))
    dates = sorted(data.portfolio_returns["date"])
    assert len(dates) == 36
    assert dates[-1] == pd.Timestamp("2025-02-28")


# --- taxonomy and benchmark ----------------------------------------------------


def test_no_sector_is_an_index_name(sample: SampleData) -> None:
    used = set(sample.benchmark["sector"]) | set(sample.holdings["sector"]) | set(sample.security_master["sector"])
    assert "Bankex" not in used
    assert not used & INDEX_NAMES_NOT_SECTORS


def test_benchmark_weights_sum_to_one_and_sit_in_bands(sample: SampleData) -> None:
    assert sum(BENCHMARK_SECTOR_WEIGHTS.values()) == pytest.approx(1.0, abs=1e-12)
    assert set(BENCHMARK_SECTOR_WEIGHTS) == set(BENCHMARK_SECTOR_BANDS) == set(SECTORS)
    for s, (lo, hi) in BENCHMARK_SECTOR_BANDS.items():
        assert lo <= BENCHMARK_SECTOR_WEIGHTS[s] <= hi, s
    bw = sample.benchmark.pivot(index="date", columns="sector", values="weight")
    np.testing.assert_allclose(bw.sum(axis=1), 1.0, atol=1e-12)
    for s, (lo, hi) in BENCHMARK_SECTOR_BANDS.items():
        assert bw[s].between(lo, hi).all(), s


def test_weights_sum_to_one_every_date(sample: SampleData) -> None:
    for frame in (sample.holdings, sample.benchmark):
        sums = frame.groupby("date")["weight"].sum()
        assert ((sums - 1.0).abs() <= 1e-9).all()


# --- portfolio realism ------------------------------------------------------


def test_active_sector_weights_respect_caps(sample: SampleData) -> None:
    worst = _active_sector_weights(sample).abs().max()
    assert (worst <= ACTIVE_CAP_WIDE).all(), worst[worst > ACTIVE_CAP_WIDE]
    wide = worst[worst > ACTIVE_CAP]
    assert len(wide) <= MAX_WIDE_SECTORS, wide


def test_no_sector_left_at_zero_unless_tiny_in_benchmark(sample: SampleData) -> None:
    h = sample.holdings.groupby(["date", "sector"])["weight"].sum().unstack(fill_value=0.0)
    bw = sample.benchmark.pivot(index="date", columns="sector", values="weight")
    h = h.reindex(columns=bw.columns, fill_value=0.0)
    zero = (h == 0.0) & (bw >= 0.02)
    assert not zero.any().any()


# --- the size tilt --------------------------------------------------------------


@pytest.fixture(scope="module")
def tilt(sample: SampleData) -> dict[str, float]:
    return tilt_diagnostics(sample)


def test_size_loading_negatively_correlated_with_log_cap(tilt: dict[str, float]) -> None:
    assert tilt["corr_size_logcap"] <= -0.60


def test_held_names_are_smaller_than_unheld(tilt: dict[str, float]) -> None:
    assert tilt["held_to_unheld_cap"] <= 0.70


def test_active_size_loading_is_positive(tilt: dict[str, float]) -> None:
    """Changed from an absolute test (portfolio >= +0.30): with sector weights now close to a
    benchmark dominated by large financials, the absolute loading is not meaningful; the
    ACTIVE size loading (portfolio minus benchmark) is what the tilt is."""
    assert tilt["portfolio_size_loading"] - tilt["benchmark_size_loading"] >= 0.30
    assert tilt["portfolio_size_loading"] > tilt["benchmark_size_loading"]


def test_persistent_small_mid_cap_tilt(sample: SampleData) -> None:
    ex = sample.true_size_exposure
    assert ((ex["portfolio"] - ex["benchmark"]) > 0.2).all()


def test_universe_size_loading_is_centred(tilt: dict[str, float]) -> None:
    assert -0.05 <= tilt["universe_size_loading"] <= 0.05


def test_active_return_in_target_band(tilt: dict[str, float]) -> None:
    assert 0.03 <= tilt["active_annualised"] <= 0.06


def test_size_explains_at_least_40pct_of_active(tilt: dict[str, float]) -> None:
    assert tilt["size_share_of_active"] >= 0.40


# --- episodes -----------------------------------------------------------------


def test_deepest_benchmark_drawdown_is_a_realistic_crisis(sample: SampleData) -> None:
    wealth = (1 + _bench_returns(sample)).cumprod()
    dd = wealth / np.maximum.accumulate(np.r_[1.0, wealth.to_numpy()])[1:] - 1
    assert -0.35 <= dd.min() <= -0.28
    trough = dd.idxmin()
    rebound = wealth.loc[trough:].iloc[1:3]
    assert (rebound > wealth.loc[trough]).all()  # partial rebound follows
    assert rebound.max() < wealth.loc[:trough].max()  # ...and it is only partial


def test_two_regime_episodes(sample: SampleData) -> None:
    r = sample.regime
    off = (r["confirmed_state"] != "normal").to_numpy()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, off.astype(int), 0])))[::2]
    assert len(runs) == 2
    crisis = r.loc[r["confirmed_state"] == "crisis"]
    assert 40 <= len(crisis) <= 70 and crisis["equity_exposure"].eq(0.6).all()
    assert (r["raw_state"] != r["confirmed_state"]).sum() > 0  # false alarms filtered out


def test_cash_and_returns_plausible(sample: SampleData) -> None:
    annual = (1 + sample.cash_returns["total_return"]) ** 12 - 1
    assert annual.between(0.05, 0.08).all()
    assert sample.prices["total_return"].abs().median() < 0.1


def test_portfolio_return_is_net_of_fees_and_exposure(sample: SampleData) -> None:
    h = sample.holdings.merge(sample.prices, on=["date", "security_id"])
    equity = (h["weight"] * h["total_return"]).groupby(h["date"]).sum()
    exposure = sample.regime.groupby(sample.regime["date"].dt.to_period("M"))["equity_exposure"].mean()
    e = equity.index.to_period("M").map(exposure).to_numpy()
    cash = sample.cash_returns.set_index("date")["total_return"]
    fee = 1.02 ** (1 / 12) - 1
    expected = e * equity + (1 - e) * cash.loc[equity.index] - fee
    actual = sample.portfolio_returns.set_index("date")["total_return"]
    np.testing.assert_allclose(actual.loc[equity.index], expected, atol=1e-12)


def test_cli(tmp_path: Path) -> None:
    assert main(["--out", str(tmp_path), "--months", "36", "--seed", "3"]) == 0
    assert (tmp_path / "holdings.csv").exists()
