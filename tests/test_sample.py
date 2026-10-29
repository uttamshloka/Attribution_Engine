from __future__ import annotations

from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from attribution import io
from attribution.sample import SampleConfig, SampleData, generate_sample, main, write_sample
from attribution.schemas import SECTORS, check_holdings_against_master

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
)


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
    data = generate_sample(SampleConfig(end_date=date(2025, 3, 14), n_months=24))
    dates = sorted(data.portfolio_returns["date"])
    assert len(dates) == 24
    assert dates[-1] == pd.Timestamp("2025-02-28")


def test_persistent_small_mid_cap_tilt(sample: SampleData) -> None:
    chars = sample.characteristics.set_index("security_id")
    size = chars["loading_size"]
    h = sample.holdings
    port_size = (h["weight"] * h["security_id"].map(size)).groupby(h["date"]).sum()

    # Benchmark is cap-weighted: rebuild security-level caps from initial caps and returns.
    prices = sample.prices.pivot(index="date", columns="security_id", values="total_return")
    caps = prices.shift(1).fillna(0.0).add(1.0).cumprod().mul(chars.loc[prices.columns, "initial_market_cap_cr"])
    bench_w = caps.div(caps.sum(axis=1), axis=0)
    bench_size = bench_w.mul(size.loc[prices.columns]).sum(axis=1)

    assert (port_size > bench_size + 0.2).all()


def test_crisis_episode(sample: SampleData) -> None:
    r = sample.regime
    crisis = (r["confirmed_state"] == "crisis").to_numpy()
    runs = np.diff(np.flatnonzero(np.diff(np.r_[0, crisis.astype(int), 0])))[::2]
    assert len(runs) == 1
    assert 55 <= runs[0] <= 65
    assert r.loc[crisis, "equity_exposure"].eq(0.6).all()
    # Raw state has false alarms that confirmation removes.
    assert (r["raw_state"] != r["confirmed_state"]).sum() > 0


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
    assert main(["--out", str(tmp_path), "--months", "24", "--seed", "3"]) == 0
    assert (tmp_path / "holdings.csv").exists()


# --- size tilt: the sample must carry a deliberate, coherent small/mid-cap tilt ---


@pytest.fixture(scope="module")
def tilt(sample: SampleData) -> dict[str, float]:
    from attribution.sample import tilt_diagnostics

    return tilt_diagnostics(sample)


def test_size_loading_negatively_correlated_with_log_cap(tilt: dict[str, float]) -> None:
    assert tilt["corr_size_logcap"] <= -0.60


def test_held_names_are_smaller_than_unheld(tilt: dict[str, float]) -> None:
    assert tilt["held_to_unheld_cap"] <= 0.70


def test_portfolio_weighted_size_loading_is_positive(tilt: dict[str, float]) -> None:
    assert tilt["portfolio_size_loading"] >= 0.30


def test_universe_size_loading_is_centred(tilt: dict[str, float]) -> None:
    assert -0.05 <= tilt["universe_size_loading"] <= 0.05


def test_active_return_in_target_band(tilt: dict[str, float]) -> None:
    assert 0.03 <= tilt["active_annualised"] <= 0.06


def test_size_explains_at_least_40pct_of_active(tilt: dict[str, float]) -> None:
    assert tilt["size_share_of_active"] >= 0.40


def test_crisis_benchmark_shape_unchanged(sample: SampleData) -> None:
    b = sample.benchmark
    monthly = (b["weight"] * b["total_return"]).groupby(b["date"]).sum()
    crisis = monthly.loc["2024-07-31":"2024-09-30"].round(4).tolist()
    assert crisis == [-0.1015, 0.0043, 0.0561]


def test_weights_sum_to_one_every_date(sample: SampleData) -> None:
    for frame in (sample.holdings, sample.benchmark):
        sums = frame.groupby("date")["weight"].sum()
        assert ((sums - 1.0).abs() <= 1e-9).all()
