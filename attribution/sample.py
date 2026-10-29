"""Synthetic sample portfolio for development and tests.

Nothing here is real market data. The generator builds a small, internally
consistent world, shaped to look like an Indian large-cap mandate, so that the
attribution maths has known answers to find.

Taxonomy
    The 13 sectors of :class:`attribution.schemas.Sector`, a condensed
    approximation of the NSE sector classification (financials of every kind
    in Financial Services; oil, gas and fuels in Oil & Gas).

Universe
    40 securities. Each sector's starting market cap is set so the benchmark
    begins at :data:`BENCHMARK_SECTOR_WEIGHTS`; within a sector, names differ
    widely in size. ISINs use the synthetic issuer prefix ``INEZZ`` and carry
    valid check digits; they are not real securities.

Benchmark
    Sector weights are reset to :data:`BENCHMARK_SECTOR_WEIGHTS` every month
    and names are cap-weighted within each sector, so the sector mix stays
    inside :data:`BENCHMARK_SECTOR_BANDS`. Those weights are APPROXIMATE and
    INDICATIVE of an Indian large-cap index, taken from no live feed; replace
    them with published index weights before any client-facing use.

Returns (monthly, simple)
    Each security earns ``cash + beta * (market - cash) + sum(loading * factor)
    + sector shock + idiosyncratic noise``. Size loading is ``-0.6`` times the
    standardised log starting cap: small names load positively on ``size``
    and the universe average is zero. The true loadings are written to
    ``sample_characteristics.csv`` and the true size exposures of portfolio and
    benchmark are kept in :attr:`SampleData.true_size_exposure`, so the factor
    view can be checked against what was put in.

Portfolio
    A concentrated but investable book: in every sector it holds all names
    except the largest, weighted towards the smallest (``cap ** -0.75``), at
    sector weights equal to the benchmark plus :data:`PORTFOLIO_ACTIVE_TILTS`.
    Active sector weights stay within +/-8 percentage points, except at most
    two sectors within +/-12. Every sector is held. Rebalanced quarterly and
    drifting with returns in between. Reported returns are net of a 2% annual
    fee; equity exposure comes from the regime series, the rest is in cash.

Market episodes
    Two drawdowns, set by solving for the market factor so the BENCHMARK hits
    the stated monthly returns:

    * a correction in 2023-12 / 2024-01 (about -10% peak to trough), during
      which the regime model goes to "elevated" (equity 0.85);
    * a crisis in 2024-08 / 2024-09 (about -30% peak to trough over two
      months), then a partial rebound in 2024-10 / 2024-11, with the regime
      model at "crisis" (equity 0.60) from early in the fall until into the
      rebound - so the overlay earns while the market falls and gives some
      back while it recovers.

    The regime series is business-daily, with occasional 1-3 day false alarms
    in the raw state that the 5-day confirmation filter removes. Small caps
    fall harder in the crisis (a size shock), then earn a deterministic
    small/mid-cap premium after it.

Cash
    A T-bill-like yield wandering between 5.5% and 7.5% a year.

Timing follows the package convention: a row dated ``t`` describes the period
ending ``t``. Weights dated ``t`` were set at the close of the previous period.

Run ``python -m attribution.sample --out sample_data`` to write the files.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from attribution.io import write_policy, write_table
from attribution.schemas import (
    BENCHMARK,
    CASH_KEY,
    CASH_RETURN,
    FACTOR_RETURNS,
    FACTORS,
    HOLDINGS,
    PORTFOLIO_RETURN,
    PRICES,
    REGIME_SERIES,
    SECTORS,
    SECURITY_MASTER,
    Policy,
    RegimeState,
    Sector,
    isin_check_digit,
)

__all__ = [
    "DEFAULT_END_DATE",
    "DEFAULT_SEED",
    "BENCHMARK_SECTOR_WEIGHTS",
    "BENCHMARK_SECTOR_BANDS",
    "PORTFOLIO_ACTIVE_TILTS",
    "ACTIVE_CAP",
    "ACTIVE_CAP_WIDE",
    "MAX_WIDE_SECTORS",
    "SampleConfig",
    "SampleData",
    "generate_sample",
    "write_sample",
    "tilt_diagnostics",
    "main",
]

DEFAULT_END_DATE = date(2026, 9, 30)
DEFAULT_SEED = 20261028

# ---------------------------------------------------------------------------
# APPROXIMATE, INDICATIVE benchmark sector weights for an Indian large-cap
# index. Sourced from NO live feed. Replace with published index weights
# before any client-facing use.
# ---------------------------------------------------------------------------
BENCHMARK_SECTOR_WEIGHTS: dict[str, float] = {
    Sector.FINANCIAL_SERVICES.value: 0.370,
    Sector.INFORMATION_TECHNOLOGY.value: 0.135,
    Sector.OIL_AND_GAS.value: 0.110,
    Sector.FMCG.value: 0.080,
    Sector.AUTO.value: 0.070,
    Sector.HEALTHCARE.value: 0.045,
    Sector.CAPITAL_GOODS.value: 0.035,
    Sector.TELECOMMUNICATION.value: 0.035,
    Sector.METAL.value: 0.030,
    Sector.POWER.value: 0.025,
    Sector.CONSUMER_DURABLES.value: 0.025,
    Sector.BASIC_MATERIALS.value: 0.025,
    Sector.REALTY.value: 0.015,
}
#: The approximate band each sector weight is meant to sit in (APPROXIMATE).
BENCHMARK_SECTOR_BANDS: dict[str, tuple[float, float]] = {
    Sector.FINANCIAL_SERVICES.value: (0.35, 0.40),
    Sector.INFORMATION_TECHNOLOGY.value: (0.12, 0.15),
    Sector.OIL_AND_GAS.value: (0.10, 0.12),
    Sector.FMCG.value: (0.07, 0.09),
    Sector.AUTO.value: (0.06, 0.08),
    Sector.HEALTHCARE.value: (0.04, 0.05),
    Sector.CAPITAL_GOODS.value: (0.03, 0.04),
    Sector.TELECOMMUNICATION.value: (0.03, 0.04),
    Sector.METAL.value: (0.02, 0.04),
    Sector.POWER.value: (0.02, 0.03),
    Sector.CONSUMER_DURABLES.value: (0.02, 0.03),
    Sector.BASIC_MATERIALS.value: (0.005, 0.04),
    Sector.REALTY.value: (0.005, 0.03),
}

#: Securities per sector (40 in all). Larger sectors have more names.
_NAMES_PER_SECTOR: dict[str, int] = {
    Sector.FINANCIAL_SERVICES.value: 6,
    Sector.INFORMATION_TECHNOLOGY.value: 4,
    Sector.OIL_AND_GAS.value: 3,
    Sector.FMCG.value: 3,
    Sector.AUTO.value: 3,
    Sector.HEALTHCARE.value: 3,
    Sector.CAPITAL_GOODS.value: 3,
    Sector.TELECOMMUNICATION.value: 2,
    Sector.METAL.value: 3,
    Sector.POWER.value: 3,
    Sector.CONSUMER_DURABLES.value: 3,
    Sector.BASIC_MATERIALS.value: 2,
    Sector.REALTY.value: 2,
}

#: Portfolio sector weight minus benchmark sector weight at each rebalance.
#: Sums to zero. Two sectors are allowed up to +/-12pp, the rest +/-8pp; the
#: values leave room for drift between quarterly rebalances.
PORTFOLIO_ACTIVE_TILTS: dict[str, float] = {
    Sector.FINANCIAL_SERVICES.value: -0.090,
    Sector.INFORMATION_TECHNOLOGY.value: -0.030,
    Sector.OIL_AND_GAS.value: -0.045,
    Sector.FMCG.value: -0.020,
    Sector.AUTO.value: +0.015,
    Sector.HEALTHCARE.value: +0.030,
    Sector.CAPITAL_GOODS.value: +0.055,
    Sector.TELECOMMUNICATION.value: -0.010,
    Sector.METAL.value: +0.010,
    Sector.POWER.value: +0.015,
    Sector.CONSUMER_DURABLES.value: +0.040,
    Sector.BASIC_MATERIALS.value: +0.015,
    Sector.REALTY.value: +0.015,
}
ACTIVE_CAP = 0.08
ACTIVE_CAP_WIDE = 0.12
MAX_WIDE_SECTORS = 2

#: (monthly mean, monthly volatility) for each factor return.
_FACTOR_MOMENTS: dict[str, tuple[float, float]] = {
    "market": (0.011, 0.045),
    "size": (0.002, 0.025),
    "value": (0.002, 0.022),
    "momentum": (0.004, 0.028),
    "quality": (0.003, 0.016),
    "low_volatility": (0.002, 0.020),
}

# Month indices (0 = first month) of the two episodes, for the default 60-month
# window ending 2026-09: correction 2023-12/2024-01, crisis 2024-08/2024-09,
# rebound 2024-10/2024-11. For other windows they scale with the length.
_CORRECTION_AT = 26 / 60
_CRISIS_AT = 34 / 60
#: Benchmark monthly returns the market factor is solved to hit.
_CORRECTION_PATH = (-0.060, -0.045)
_CRISIS_PATH = (-0.170, -0.150, +0.080, +0.055)
#: The month before each episode is pinned to a small gain, so the peak sits
#: just before the fall and the drawdown depth is set by the path above.
_PRE_EPISODE_RETURN = 0.015
#: Extra shocks to style factors in the four crisis months.
_CRISIS_STYLE_SHOCKS: dict[str, tuple[float, ...]] = {
    "size": (-0.030, -0.025, +0.020, +0.010),
    "momentum": (-0.030, -0.010, -0.020, 0.000),
    "low_volatility": (+0.020, +0.015, -0.010, 0.000),
    "quality": (+0.015, +0.010, 0.000, 0.000),
}
#: Small/mid-cap premium added to the size factor every month after the crisis.
_SIZE_PREMIUM_MONTHLY = 0.018

_EXPOSURE_BY_STATE: dict[str, float] = {
    RegimeState.NORMAL.value: 1.00,
    RegimeState.ELEVATED.value: 0.85,
    RegimeState.CRISIS.value: 0.60,
}
_CONFIRM_DAYS = 5
_ANNUAL_FEE = 0.02
_REBALANCE_EVERY = 3
_CAP_EXPONENT = -0.75
_UNIVERSE_CAP_CR = 2_000_000.0
_BENCHMARK_NAME = "Synthetic Indian large-cap (sector-weighted, indicative)"


@dataclass(frozen=True)
class SampleConfig:
    end_date: date = DEFAULT_END_DATE
    n_months: int = 60
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.n_months < 24:
            raise ValueError("n_months must be at least 24 so both episodes fit inside the history")
        if not isinstance(self.end_date, date):
            raise TypeError("end_date must be a datetime.date")


@dataclass(frozen=True)
class SampleData:
    """All generated tables, already validated against their schemas."""

    config: SampleConfig
    security_master: pd.DataFrame
    holdings: pd.DataFrame
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    portfolio_returns: pd.DataFrame
    factor_returns: pd.DataFrame
    cash_returns: pd.DataFrame
    regime: pd.DataFrame
    policy: Policy
    characteristics: pd.DataFrame = field(repr=False)
    true_size_exposure: pd.DataFrame = field(repr=False)

    def header(self) -> list[str]:
        return [
            "SYNTHETIC SAMPLE DATA - generated by attribution.sample. Not real market data.",
            f"seed={self.config.seed} end_date={self.config.end_date.isoformat()} "
            f"n_months={self.config.n_months}",
            "Securities, ISINs, prices, returns and benchmark weights are invented or indicative. "
            "Do not use for any investment decision.",
        ]


def _month_ends(end: date, n: int) -> pd.DatetimeIndex:
    """``n`` month-end dates, the last being the month end on or before ``end``."""
    end_ts = pd.Timestamp(end)
    last = end_ts if end_ts.is_month_end else end_ts - pd.offsets.MonthEnd(1)
    return pd.date_range(end=last.normalize(), periods=n, freq="ME").as_unit("ns")


def _make_isin(index: int) -> str:
    body = f"INEZZ{index:02d}A010"
    return body + str(isin_check_digit(body))


def _security_master() -> tuple[pd.DataFrame, list[str]]:
    rows = []
    sectors_by_security: list[str] = []
    n = 0
    for sector in Sector:
        for k in range(1, _NAMES_PER_SECTOR[sector.value] + 1):
            n += 1
            rows.append({
                "security_id": f"SYN{n:03d}",
                "name": f"Synthetic {sector.value} {k}",
                "sector": sector.value,
                "isin": _make_isin(n),
            })
            sectors_by_security.append(sector.value)
    master = pd.DataFrame(rows)
    assert len(master) == 40
    return master, sectors_by_security


def _episode_months(T: int) -> tuple[list[int], list[int]]:
    c0 = int(round(_CORRECTION_AT * T))
    k0 = int(round(_CRISIS_AT * T))
    correction = [c0, c0 + 1]
    crisis = [k0 + i for i in range(len(_CRISIS_PATH))]
    if crisis[-1] >= T - 1:
        raise ValueError("window too short for the crisis and its rebound")
    return correction, crisis


def _regime(rng: np.random.Generator, months: pd.DatetimeIndex, correction: list[int], crisis: list[int]) -> pd.DataFrame:
    start = months[0] - pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    days = pd.bdate_range(start=start, end=months[-1]).as_unit("ns")
    n = len(days)
    raw = np.full(n, RegimeState.NORMAL.value, dtype=object)
    month_of_day = np.searchsorted(months.values, days.values)  # day -> index of its month end

    def first_day(m: int) -> int:
        return int(np.flatnonzero(month_of_day == m)[0])

    # Correction: elevated from ~6th business day of the first month to ~12th of the second.
    a = first_day(correction[0]) + 5
    b = first_day(correction[1]) + 12
    raw[a:b] = RegimeState.ELEVATED.value
    # Crisis: elevated for 3 days early in the fall, crisis from ~5th day of the fall
    # through the first rebound month, then elevated ~10 days into the second rebound month.
    c = first_day(crisis[0]) + 1
    raw[c:c + 3] = RegimeState.ELEVATED.value
    d = c + 3
    e = first_day(crisis[3]) + 2
    raw[d:e] = RegimeState.CRISIS.value
    raw[e:e + 10] = RegimeState.ELEVATED.value

    # False alarms: short raw blips well away from both episodes.
    busy = np.zeros(n, dtype=bool)
    busy[max(0, a - 30):b + 30] = True
    busy[max(0, c - 30):e + 40] = True
    quiet = np.flatnonzero(~busy[: n - 10])
    quiet = quiet[quiet > 30]
    for s in rng.choice(quiet, size=6, replace=False):
        raw[s:s + int(rng.integers(1, 4))] = RegimeState.ELEVATED.value

    confirmed = np.empty(n, dtype=object)
    current = raw[0]
    run = 0
    for i in range(n):
        run = run + 1 if i > 0 and raw[i] == raw[i - 1] else 1
        if raw[i] != current and run >= _CONFIRM_DAYS:
            current = raw[i]
        confirmed[i] = current
    exposure = np.array([_EXPOSURE_BY_STATE[s] for s in confirmed])
    return pd.DataFrame({"date": days, "raw_state": raw, "confirmed_state": confirmed, "equity_exposure": exposure})


def generate_sample(config: SampleConfig | None = None) -> SampleData:
    """Generate the full sample world. Same config gives identical output."""
    cfg = config or SampleConfig()
    rng = np.random.default_rng(cfg.seed)
    months = _month_ends(cfg.end_date, cfg.n_months)
    T = len(months)
    correction, crisis = _episode_months(T)

    master, sector_of = _security_master()
    n_sec = len(master)
    sector_arr = np.array(sector_of)
    bw = np.array([BENCHMARK_SECTOR_WEIGHTS[s] for s in SECTORS])
    assert abs(bw.sum() - 1.0) < 1e-12

    # --- starting caps: sector totals match the benchmark weights ------------
    cap0 = np.empty(n_sec)
    for s in SECTORS:
        idx = np.flatnonzero(sector_arr == s)
        shares = np.sort(np.exp(rng.normal(0.0, 1.0, idx.size)))[::-1]
        cap0[idx] = BENCHMARK_SECTOR_WEIGHTS[s] * _UNIVERSE_CAP_CR * shares / shares.sum()
    log_cap0 = np.log(cap0)
    z_size = (log_cap0 - log_cap0.mean()) / log_cap0.std()
    loadings = {
        "size": -0.6 * z_size,
        "value": rng.normal(0.0, 0.5, n_sec),
        "momentum": rng.normal(0.0, 0.4, n_sec),
        "quality": rng.normal(0.0, 0.4, n_sec),
        "low_volatility": rng.normal(0.0, 0.4, n_sec),
    }
    beta = np.clip(rng.normal(1.0, 0.15, n_sec) + 0.06 * (-z_size), 0.6, 1.5)
    idio_vol = 0.030 + 0.015 * np.clip(-z_size, 0.0, None)

    # --- regime & cash ---------------------------------------------------------
    regime = _regime(rng, months, correction, crisis)
    month_key = regime["date"].dt.to_period("M")
    monthly_exposure = regime.groupby(month_key)["equity_exposure"].mean()
    exposure_by_month = np.array([monthly_exposure[m.to_period("M")] for m in months])

    annual_yield = np.empty(T)
    y = 0.066
    for t in range(T):
        y = float(np.clip(y + rng.normal(0.0, 0.0015), 0.055, 0.075))
        annual_yield[t] = y
    cash = (1.0 + annual_yield) ** (1.0 / 12.0) - 1.0

    # --- random draws (all drawn up front, so the stream never depends on solving) ---
    factors = {f: rng.normal(mu, sd, T) for f, (mu, sd) in _FACTOR_MOMENTS.items()}
    sector_shock = {s: rng.normal(0.0, 0.020, T) for s in SECTORS}
    idio = rng.normal(0.0, 1.0, (T, n_sec)) * idio_vol
    for f, shocks in _CRISIS_STYLE_SHOCKS.items():
        for j, t in enumerate(crisis):
            factors[f][t] += shocks[j]
    for t in range(crisis[-1] + 1, T):
        factors["size"][t] += _SIZE_PREMIUM_MONTHLY
    targets = (dict(zip(correction, _CORRECTION_PATH, strict=True)) | dict(zip(crisis, _CRISIS_PATH, strict=True))
               | {correction[0] - 1: _PRE_EPISODE_RETURN, crisis[0] - 1: _PRE_EPISODE_RETURN})

    # --- security returns, benchmark weights and caps, month by month ---------
    sec_returns = np.empty((T, n_sec))
    caps = np.empty((T, n_sec))
    bench_w = np.empty((T, n_sec))
    caps[0] = cap0
    sector_mask = {s: sector_arr == s for s in SECTORS}
    for t in range(T):
        if t > 0:
            caps[t] = caps[t - 1] * (1.0 + sec_returns[t - 1])
        w = np.empty(n_sec)
        for s in SECTORS:
            m = sector_mask[s]
            w[m] = BENCHMARK_SECTOR_WEIGHTS[s] * caps[t, m] / caps[t, m].sum()
        bench_w[t] = w
        rest = cash[t] * (1.0 - beta) + idio[t] + np.array([sector_shock[s][t] for s in sector_arr])
        for f, load in loadings.items():
            rest = rest + load * factors[f][t]
        if t in targets:  # solve the market factor so the benchmark hits the target exactly
            factors["market"][t] = (targets[t] - float(w @ rest)) / float(w @ beta)
        sec_returns[t] = np.maximum(rest + beta * factors["market"][t], -0.9)

    # --- benchmark (sector level) -------------------------------------------------
    bench_rows = []
    for t in range(T):
        for s in SECTORS:
            m = sector_mask[s]
            ws = float(bench_w[t, m].sum())
            rs = float((bench_w[t, m] * sec_returns[t, m]).sum() / ws)
            bench_rows.append({"date": months[t], "sector": s, "weight": ws, "total_return": rs})
    benchmark = pd.DataFrame(bench_rows)
    benchmark["weight"] = benchmark["weight"] / benchmark.groupby("date")["weight"].transform("sum")

    # --- portfolio ------------------------------------------------------------------
    held = np.ones(n_sec, dtype=bool)
    for s in SECTORS:
        idx = np.flatnonzero(sector_mask[s])
        held[idx[np.argmax(cap0[idx])]] = False  # leave out the largest name in every sector
    port_sector = {s: BENCHMARK_SECTOR_WEIGHTS[s] + PORTFOLIO_ACTIVE_TILTS[s] for s in SECTORS}
    assert abs(sum(port_sector.values()) - 1.0) < 1e-12 and min(port_sector.values()) > 0
    weights = np.empty((T, n_sec))
    for t in range(T):
        if t % _REBALANCE_EVERY == 0:
            target = np.zeros(n_sec)
            for s in SECTORS:
                m = sector_mask[s] & held
                raw_w = caps[t, m] ** _CAP_EXPONENT
                target[m] = port_sector[s] * raw_w / raw_w.sum()
            weights[t] = target / target.sum()
        else:
            grown = weights[t - 1] * (1.0 + sec_returns[t - 1])
            weights[t] = grown / grown.sum()

    holdings_rows = []
    for t in range(T):
        for i in np.flatnonzero(weights[t] > 0):
            holdings_rows.append({"date": months[t], "security_id": master.at[i, "security_id"],
                                  "weight": float(weights[t, i]), "sector": sector_arr[i]})
    holdings = pd.DataFrame(holdings_rows)

    equity_return = (weights * sec_returns).sum(axis=1)
    fee = (1.0 + _ANNUAL_FEE) ** (1.0 / 12.0) - 1.0
    portfolio_net = exposure_by_month * equity_return + (1.0 - exposure_by_month) * cash - fee

    prices = pd.DataFrame({
        "date": np.repeat(months.values, n_sec),
        "security_id": np.tile(master["security_id"].to_numpy(), T),
        "total_return": sec_returns.reshape(-1),
    })
    portfolio_returns = pd.DataFrame({"date": months, "total_return": portfolio_net})
    factor_returns = pd.DataFrame({
        "date": np.repeat(months.values, len(FACTORS)),
        "factor": np.tile(np.array(FACTORS, dtype=object), T),
        "total_return": np.column_stack([factors[f] for f in FACTORS]).reshape(-1),
    })
    cash_returns = pd.DataFrame({"date": months, "total_return": cash})

    # --- policy: neutral = benchmark sector weights, 100% equity ---------------------
    neutral = {s: round(BENCHMARK_SECTOR_WEIGHTS[s], 4) for s in SECTORS}
    neutral[CASH_KEY] = 0.0
    bands = {s: (round(max(0.0, w - 0.05), 4), round(min(1.0, w + 0.05), 4)) for s, w in neutral.items() if s != CASH_KEY}
    bands[CASH_KEY] = (0.0, 0.40)
    policy = Policy(neutral_weights=neutral, bands=bands, derisk_mode="cash",
                    review_date=months[-1].date(), benchmark_name=_BENCHMARK_NAME)

    characteristics = pd.DataFrame({
        "security_id": master["security_id"],
        "sector": master["sector"],
        "held": held,
        "initial_market_cap_cr": np.round(cap0, 2),
        "beta": beta,
        **{f"loading_{f}": v for f, v in loadings.items()},
        "idiosyncratic_vol": idio_vol,
    })
    true_size = pd.DataFrame({
        "date": months,
        "portfolio": weights @ loadings["size"],
        "benchmark": bench_w @ loadings["size"],
        "equity_exposure": exposure_by_month,
    })

    return SampleData(
        config=cfg,
        security_master=SECURITY_MASTER.validate(master, source="sample.security_master"),
        holdings=HOLDINGS.validate(holdings, source="sample.holdings"),
        prices=PRICES.validate(prices, source="sample.prices"),
        benchmark=BENCHMARK.validate(benchmark, source="sample.benchmark"),
        portfolio_returns=PORTFOLIO_RETURN.validate(portfolio_returns, source="sample.portfolio_returns"),
        factor_returns=FACTOR_RETURNS.validate(factor_returns, source="sample.factor_returns"),
        cash_returns=CASH_RETURN.validate(cash_returns, source="sample.cash_returns"),
        regime=REGIME_SERIES.validate(regime, source="sample.regime"),
        policy=policy,
        characteristics=characteristics,
        true_size_exposure=true_size,
    )


def tilt_diagnostics(data: SampleData) -> dict[str, float]:
    """Measures that show whether the sample carries its intended size tilt.

    * ``corr_size_logcap``: correlation of ``loading_size`` with log starting cap.
    * ``held_to_unheld_cap``: mean starting cap of held names / of unheld names.
    * ``portfolio_size_loading``: weighted size loading of the equity sleeve,
      averaged across dates.
    * ``benchmark_size_loading``: the benchmark's, averaged across dates.
    * ``universe_size_loading``: simple average size loading of all securities.
    * ``active_annualised``: annualised net portfolio return minus annualised
      benchmark return (benchmark = ``sum(weight * return)`` over sectors).
    * ``size_contribution_annualised``: sum over months of (portfolio size
      exposure x equity exposure - benchmark size exposure) x size factor
      return, per year.
    * ``size_share_of_active``: the ratio of the two.
    """
    chars = data.characteristics.set_index("security_id")
    loading = chars["loading_size"]
    cap0 = chars["initial_market_cap_cr"]
    held = chars["held"].astype(bool)
    ex = data.true_size_exposure.set_index("date")
    years = len(ex) / 12.0
    port = data.portfolio_returns.set_index("date")["total_return"]
    b = data.benchmark
    bench = (b["weight"] * b["total_return"]).groupby(b["date"]).sum()
    active = float((1 + port).prod() ** (1 / years) - (1 + bench).prod() ** (1 / years))
    fr = data.factor_returns
    size_ret = fr.loc[fr["factor"] == "size"].set_index("date")["total_return"]
    active_exp = ex["portfolio"] * ex["equity_exposure"] - ex["benchmark"]
    size_contrib = float((active_exp * size_ret.loc[ex.index]).sum() / years)
    return {
        "corr_size_logcap": float(np.corrcoef(loading, np.log(cap0))[0, 1]),
        "held_to_unheld_cap": float(cap0[held].mean() / cap0[~held].mean()),
        "portfolio_size_loading": float(ex["portfolio"].mean()),
        "benchmark_size_loading": float(ex["benchmark"].mean()),
        "universe_size_loading": float(loading.mean()),
        "active_annualised": active,
        "size_contribution_annualised": size_contrib,
        "size_share_of_active": size_contrib / active if active != 0 else float("nan"),
    }


#: File stem written for each table.
FILE_STEMS: dict[str, str] = {
    "security_master": "security_master",
    "holdings": "holdings",
    "prices": "prices",
    "benchmark": "benchmark",
    "portfolio_returns": "portfolio_returns",
    "factor_returns": "factor_returns",
    "cash_returns": "cash_returns",
    "regime": "regime_series",
}

_SCHEMA_FOR = {
    "security_master": SECURITY_MASTER,
    "holdings": HOLDINGS,
    "prices": PRICES,
    "benchmark": BENCHMARK,
    "portfolio_returns": PORTFOLIO_RETURN,
    "factor_returns": FACTOR_RETURNS,
    "cash_returns": CASH_RETURN,
    "regime": REGIME_SERIES,
}


def write_sample(data: SampleData, out_dir: str | Path, fmt: str = "csv") -> dict[str, Path]:
    """Write every table (``csv`` or ``parquet``), the policy JSON and the characteristics CSV."""
    if fmt not in ("csv", "parquet"):
        raise ValueError(f"fmt must be 'csv' or 'parquet', got {fmt!r}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = data.header()
    written: dict[str, Path] = {}
    for attr, stem in FILE_STEMS.items():
        written[attr] = write_table(getattr(data, attr), out / f"{stem}.{fmt}", _SCHEMA_FOR[attr], header)
    written["policy"] = write_policy(data.policy, out / "policy.json", header)

    chars_path = out / "sample_characteristics.csv"
    chars_header = header + [
        "Ground-truth loadings used to generate returns. Not an input contract; provided to check the factor view."
    ]
    body = data.characteristics.to_csv(index=False, lineterminator="\n", float_format="%.12g")
    chars_path.write_text("".join(f"# {line}\n" for line in chars_header) + body, encoding="utf-8")
    written["characteristics"] = chars_path
    return written


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write the synthetic sample portfolio.")
    parser.add_argument("--out", type=Path, default=Path("sample_data"), help="output directory")
    parser.add_argument("--end", type=date.fromisoformat, default=DEFAULT_END_DATE, help="end date YYYY-MM-DD")
    parser.add_argument("--months", type=int, default=60, help="months of history")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--format", choices=("csv", "parquet"), default="csv")
    args = parser.parse_args(argv)
    data = generate_sample(SampleConfig(end_date=args.end, n_months=args.months, seed=args.seed))
    paths = write_sample(data, args.out, args.format)
    for name, path in paths.items():
        print(f"{name:18s} {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
