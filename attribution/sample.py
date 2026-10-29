"""Synthetic sample portfolio for development and tests.

Nothing here is real market data. The generator builds a small, internally
consistent world so that later attribution maths has known answers to find:

Universe
    40 securities spread over all 15 BSE sectors (3 in each of ten larger
    sectors, 2 in each of five smaller ones), with log-normal market caps.
    ISINs use the synthetic issuer prefix ``INEZZ`` and carry valid check
    digits; they are not real securities.

Returns (monthly, simple)
    Each security earns ``cash + beta * (market - cash) + sum(loading * factor)
    + sector shock + idiosyncratic noise``. Size loading is ``-0.6`` times the
    standardised log starting cap, so it is exactly negatively correlated with
    log cap: small caps load positively on ``size``, and the universe average
    is zero. Because caps are log-normal, a few giants dominate any
    arithmetic-mean cap; judge the link on log cap. The other style loadings
    are random. After the crisis the size factor earns a deterministic premium
    of ``_SIZE_PREMIUM_MONTHLY`` a month (the small/mid-cap recovery); it starts
    after the last crisis month so nothing up to the crisis changes. The true loadings are written to
    ``sample_characteristics.csv`` so a later factor view can be checked
    against them; that file is *not* an input contract.

Benchmark
    Cap-weighted over all 40 securities; sector weights are beginning-of-period
    cap shares and sector returns are cap-weighted within the sector.

Portfolio
    Holds the 32 smallest of the 40 securities by starting cap (the 8 largest
    are excluded) with weights proportional to ``cap ** _CAP_EXPONENT``
    (negative, so smaller names weigh more) times fixed sector tilts. Selection
    and weighting both lean small/mid cap, giving a persistent positive size
    exposure against a benchmark whose cap weighting leans large.
    Rebalanced quarterly, drifting with returns in between. Its equity sleeve
    weights (``holdings``) sum to 1 per date; the portfolio's total exposure to
    equity comes from the regime series, with the rest in cash. Reported
    portfolio returns are net of a 2% annual management fee.

Regime
    Business-daily series with one crisis episode of 60 business days in the
    middle of the history, a short elevated run either side, and occasional
    1-3 day false alarms in the raw state that the 5-day confirmation filter
    removes. Equity exposure follows the confirmed state: normal 1.00,
    elevated 0.85, crisis 0.60. The market factor takes a drawdown in the
    months the crisis covers.

Cash
    A T-bill-like yield wandering between 5.5% and 7.5% a year, compounded to
    monthly.

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
    "SampleConfig",
    "SampleData",
    "generate_sample",
    "write_sample",
    "tilt_diagnostics",
    "main",
]

DEFAULT_END_DATE = date(2026, 9, 30)
DEFAULT_SEED = 20261028

#: Sectors that get three securities; the other five get two.
_LARGE_SECTORS: tuple[Sector, ...] = (
    Sector.BANKEX,
    Sector.INFORMATION_TECHNOLOGY,
    Sector.FINANCIAL_SERVICES,
    Sector.OIL_AND_GAS,
    Sector.FMCG,
    Sector.AUTO,
    Sector.HEALTHCARE,
    Sector.CAPITAL_GOODS,
    Sector.METAL,
    Sector.POWER,
)

#: Fixed sector tilts in the portfolio (multipliers on the size-tilted weight).
_SECTOR_TILT: dict[str, float] = {
    Sector.CAPITAL_GOODS.value: 1.5,
    Sector.CONSUMER_DURABLES.value: 1.4,
    Sector.HEALTHCARE.value: 1.3,
    Sector.BANKEX.value: 0.7,
    Sector.OIL_AND_GAS.value: 0.6,
    Sector.METAL.value: 0.8,
}

#: (monthly mean, monthly volatility) for each factor return.
_FACTOR_MOMENTS: dict[str, tuple[float, float]] = {
    "market": (0.011, 0.050),
    "size": (0.002, 0.028),
    "value": (0.002, 0.024),
    "momentum": (0.005, 0.032),
    "quality": (0.003, 0.018),
    "low_volatility": (0.002, 0.022),
}

#: Shocks added to factor returns in the first three months touched by the crisis.
_CRISIS_SHOCKS: dict[str, tuple[float, ...]] = {
    "market": (-0.11, -0.07, 0.05),
    "size": (-0.035, -0.02, 0.02),
    "momentum": (-0.03, 0.0, -0.02),
    "low_volatility": (0.02, 0.015, -0.01),
    "quality": (0.015, 0.01, 0.0),
}

_EXPOSURE_BY_STATE: dict[str, float] = {
    RegimeState.NORMAL.value: 1.00,
    RegimeState.ELEVATED.value: 0.85,
    RegimeState.CRISIS.value: 0.60,
}

_CONFIRM_DAYS = 5
_CRISIS_DAYS = 60
_ANNUAL_FEE = 0.02
_N_HELD = 32
_CAP_EXPONENT = -0.75
_SIZE_PREMIUM_MONTHLY = 0.013
_REBALANCE_EVERY = 3
_BENCHMARK_NAME = "Synthetic BSE 40 (cap-weighted)"


@dataclass(frozen=True)
class SampleConfig:
    end_date: date = DEFAULT_END_DATE
    n_months: int = 60
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.n_months < 12:
            raise ValueError("n_months must be at least 12 so the crisis fits inside the history")
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

    def header(self) -> list[str]:
        return [
            "SYNTHETIC SAMPLE DATA - generated by attribution.sample. Not real market data.",
            f"seed={self.config.seed} end_date={self.config.end_date.isoformat()} "
            f"n_months={self.config.n_months}",
            "Securities, ISINs, prices and returns are invented. Do not use for any investment decision.",
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
        count = 3 if sector in _LARGE_SECTORS else 2
        for k in range(1, count + 1):
            n += 1
            rows.append(
                {
                    "security_id": f"SYN{n:03d}",
                    "name": f"Synthetic {sector.value} {k}",
                    "sector": sector.value,
                    "isin": _make_isin(n),
                }
            )
            sectors_by_security.append(sector.value)
    master = pd.DataFrame(rows)
    assert len(master) == 40
    return master, sectors_by_security


def _regime(rng: np.random.Generator, months: pd.DatetimeIndex) -> pd.DataFrame:
    start = months[0] - pd.offsets.MonthEnd(1) + pd.Timedelta(days=1)
    days = pd.bdate_range(start=start, end=months[-1]).as_unit("ns")
    n = len(days)
    raw = np.full(n, RegimeState.NORMAL.value, dtype=object)

    # Crisis placed a little past the middle of the history.
    crisis_start = int(n * 0.55)
    pre, post = 10, 8
    raw[crisis_start - pre : crisis_start] = RegimeState.ELEVATED.value
    raw[crisis_start : crisis_start + _CRISIS_DAYS] = RegimeState.CRISIS.value
    raw[crisis_start + _CRISIS_DAYS : crisis_start + _CRISIS_DAYS + post] = RegimeState.ELEVATED.value

    # False alarms: short raw blips well away from the crisis.
    quiet = np.r_[30 : crisis_start - pre - 30, crisis_start + _CRISIS_DAYS + post + 30 : n - 10]
    for s in rng.choice(quiet, size=6, replace=False):
        raw[s : s + int(rng.integers(1, 4))] = RegimeState.ELEVATED.value

    # Confirmation: switch only after the raw state has held for _CONFIRM_DAYS days.
    confirmed = np.empty(n, dtype=object)
    current = raw[0]
    run = 0
    for i in range(n):
        run = run + 1 if i > 0 and raw[i] == raw[i - 1] else 1
        if raw[i] != current and run >= _CONFIRM_DAYS:
            current = raw[i]
        confirmed[i] = current

    exposure = np.array([_EXPOSURE_BY_STATE[s] for s in confirmed])
    return pd.DataFrame(
        {"date": days, "raw_state": raw, "confirmed_state": confirmed, "equity_exposure": exposure}
    )


def generate_sample(config: SampleConfig | None = None) -> SampleData:
    """Generate the full sample world. Same config gives identical output."""
    cfg = config or SampleConfig()
    rng = np.random.default_rng(cfg.seed)
    months = _month_ends(cfg.end_date, cfg.n_months)
    T = len(months)

    master, sector_of = _security_master()
    n_sec = len(master)
    sector_arr = np.array(sector_of)

    # --- characteristics ------------------------------------------------
    log_cap0 = rng.normal(11.0, 1.1, size=n_sec)  # INR crore, log scale
    z_size = (log_cap0 - log_cap0.mean()) / log_cap0.std()
    loadings = {
        "size": -0.6 * z_size,
        "value": rng.normal(0.0, 0.5, n_sec),
        "momentum": rng.normal(0.0, 0.4, n_sec),
        "quality": rng.normal(0.0, 0.4, n_sec),
        "low_volatility": rng.normal(0.0, 0.4, n_sec),
    }
    beta = np.clip(rng.normal(1.0, 0.18, n_sec) + 0.08 * (-z_size), 0.6, 1.5)
    idio_vol = 0.035 + 0.02 * np.clip(-z_size, 0.0, None)

    # --- regime & cash --------------------------------------------------
    regime = _regime(rng, months)
    month_key = regime["date"].dt.to_period("M")
    monthly_exposure = regime.groupby(month_key)["equity_exposure"].mean()
    exposure_by_month = np.array([monthly_exposure[m.to_period("M")] for m in months])
    crisis_days = regime.loc[regime["confirmed_state"] == RegimeState.CRISIS.value, "date"]
    crisis_months = sorted({d.to_period("M") for d in crisis_days})
    crisis_idx = [i for i, m in enumerate(months) if m.to_period("M") in crisis_months[:3]]

    annual_yield = np.empty(T)
    y = 0.066
    for t in range(T):
        y = float(np.clip(y + rng.normal(0.0, 0.0015), 0.055, 0.075))
        annual_yield[t] = y
    cash = (1.0 + annual_yield) ** (1.0 / 12.0) - 1.0

    # --- factor returns -------------------------------------------------
    factors = {f: rng.normal(mu, sd, T) for f, (mu, sd) in _FACTOR_MOMENTS.items()}
    for f, shocks in _CRISIS_SHOCKS.items():
        for j, t in enumerate(crisis_idx):
            factors[f][t] += shocks[j]
    # Small/mid-cap premium in the recovery after the crisis. It is deterministic
    # (draws no random numbers) and starts only after the last crisis month, so
    # every value up to and including the crisis is unchanged by it.
    if crisis_idx:
        for t in range(crisis_idx[-1] + 1, T):
            factors["size"][t] += _SIZE_PREMIUM_MONTHLY

    # --- security returns -----------------------------------------------
    sector_shock = {s: rng.normal(0.0, 0.025, T) for s in SECTORS}
    sec_returns = np.empty((T, n_sec))
    for t in range(T):
        r = cash[t] + beta * (factors["market"][t] - cash[t])
        for f, load in loadings.items():
            r = r + load * factors[f][t]
        r = r + np.array([sector_shock[s][t] for s in sector_arr])
        r = r + rng.normal(0.0, 1.0, n_sec) * idio_vol
        sec_returns[t] = np.maximum(r, -0.9)

    # Caps at the start of each period, evolving with returns.
    caps = np.empty((T, n_sec))
    caps[0] = np.exp(log_cap0)
    for t in range(1, T):
        caps[t] = caps[t - 1] * (1.0 + sec_returns[t - 1])

    # --- benchmark (sector level) ---------------------------------------
    bench_rows = []
    for t in range(T):
        w = caps[t] / caps[t].sum()
        for s in SECTORS:
            mask = sector_arr == s
            ws = w[mask].sum()
            rs = float((w[mask] * sec_returns[t, mask]).sum() / ws)
            bench_rows.append({"date": months[t], "sector": s, "weight": ws, "total_return": rs})
    benchmark = pd.DataFrame(bench_rows)
    # Normalise once more so float noise cannot breach the 1e-6 contract.
    benchmark["weight"] = benchmark["weight"] / benchmark.groupby("date")["weight"].transform("sum")

    # --- portfolio ------------------------------------------------------
    # Selection: drop the largest names by starting cap, hold the rest.
    # Weighting: proportional to cap ** _CAP_EXPONENT (negative), so the
    # smallest held names get the most weight. Selection and weighting both
    # push toward small/mid caps, and the tilt is reset at every rebalance.
    by_cap = np.argsort(-log_cap0)
    held = np.ones(n_sec, dtype=bool)
    held[by_cap[: n_sec - _N_HELD]] = False
    tilt = np.array([_SECTOR_TILT.get(s, 1.0) for s in sector_arr])

    weights = np.empty((T, n_sec))
    for t in range(T):
        if t % _REBALANCE_EVERY == 0:
            target = np.where(held, caps[t] ** _CAP_EXPONENT * tilt, 0.0)
            weights[t] = target / target.sum()
        else:
            grown = weights[t - 1] * (1.0 + sec_returns[t - 1])
            weights[t] = grown / grown.sum()

    holdings_rows = []
    for t in range(T):
        for i in np.flatnonzero(weights[t] > 0):
            holdings_rows.append(
                {
                    "date": months[t],
                    "security_id": master.at[i, "security_id"],
                    "weight": float(weights[t, i]),
                    "sector": sector_arr[i],
                }
            )
    holdings = pd.DataFrame(holdings_rows)

    equity_return = (weights * sec_returns).sum(axis=1)
    fee = (1.0 + _ANNUAL_FEE) ** (1.0 / 12.0) - 1.0
    portfolio_net = exposure_by_month * equity_return + (1.0 - exposure_by_month) * cash - fee

    prices = pd.DataFrame(
        {
            "date": np.repeat(months.values, n_sec),
            "security_id": np.tile(master["security_id"].to_numpy(), T),
            "total_return": sec_returns.reshape(-1),
        }
    )
    portfolio_returns = pd.DataFrame({"date": months, "total_return": portfolio_net})
    factor_returns = pd.DataFrame(
        {
            "date": np.repeat(months.values, len(FACTORS)),
            "factor": np.tile(np.array(FACTORS, dtype=object), T),
            "total_return": np.column_stack([factors[f] for f in FACTORS]).reshape(-1),
        }
    )
    cash_returns = pd.DataFrame({"date": months, "total_return": cash})

    # --- policy ---------------------------------------------------------
    first = benchmark[benchmark["date"] == months[0]].set_index("sector")["weight"]
    neutral = {s: round(float(first[s]), 4) for s in SECTORS}
    drift = round(1.0 - sum(neutral.values()), 4)
    largest = max(neutral, key=lambda k: neutral[k])
    neutral[largest] = round(neutral[largest] + drift, 4)
    neutral[CASH_KEY] = 0.0
    bands = {s: (round(max(0.0, w - 0.05), 4), round(min(1.0, w + 0.05), 4)) for s, w in neutral.items() if s != CASH_KEY}
    bands[CASH_KEY] = (0.0, 0.40)
    policy = Policy(
        neutral_weights=neutral,
        bands=bands,
        derisk_mode="cash",
        review_date=months[-1].date(),
        benchmark_name=_BENCHMARK_NAME,
    )

    characteristics = pd.DataFrame(
        {
            "security_id": master["security_id"],
            "sector": master["sector"],
            "held": held,
            "initial_market_cap_cr": np.round(np.exp(log_cap0), 2),
            "beta": beta,
            **{f"loading_{f}": v for f, v in loadings.items()},
            "idiosyncratic_vol": idio_vol,
        }
    )

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
    )


def tilt_diagnostics(data: SampleData) -> dict[str, float]:
    """Measures that show whether the sample carries its intended size tilt.

    * ``corr_size_logcap``: correlation of ``loading_size`` with log starting
      cap across all securities.
    * ``held_to_unheld_cap``: mean starting cap of held names / of unheld names.
    * ``portfolio_size_loading``: weighted size loading of the equity sleeve,
      averaged across dates.
    * ``universe_size_loading``: simple average size loading of all securities.
    * ``active_annualised``: annualised portfolio return (net) minus annualised
      benchmark return, the benchmark being ``sum(weight * return)`` over
      sectors each month. Decimals: 0.04 is 4 percentage points.
    * ``size_contribution_annualised``: sum over months of (portfolio size
      exposure - benchmark size exposure) x size factor return, divided by the
      number of years. Portfolio exposure is scaled by that month's equity
      exposure; benchmark exposure uses cap weights rebuilt from prices.
    * ``size_share_of_active``: the ratio of the two.
    """
    chars = data.characteristics.set_index("security_id")
    loading = chars["loading_size"]
    cap0 = chars["initial_market_cap_cr"]
    held = chars["held"].astype(bool)

    h = data.holdings
    port_exp = (h["weight"] * h["security_id"].map(loading)).groupby(h["date"]).sum()

    prices = data.prices.pivot(index="date", columns="security_id", values="total_return")
    caps = prices.shift(1).fillna(0.0).add(1.0).cumprod().mul(cap0.loc[prices.columns])
    bench_exp = caps.div(caps.sum(axis=1), axis=0).mul(loading.loc[prices.columns]).sum(axis=1)

    regime = data.regime
    monthly_e = regime.groupby(regime["date"].dt.to_period("M"))["equity_exposure"].mean()
    equity_e = pd.Series(port_exp.index.to_period("M").map(monthly_e).to_numpy(), index=port_exp.index)

    years = len(port_exp) / 12.0
    port = data.portfolio_returns.set_index("date")["total_return"]
    b = data.benchmark
    bench = (b["weight"] * b["total_return"]).groupby(b["date"]).sum()
    active = float((1 + port).prod() ** (1 / years) - (1 + bench).prod() ** (1 / years))

    fr = data.factor_returns
    size_ret = fr.loc[fr["factor"] == "size"].set_index("date")["total_return"]
    size_contrib = float(((equity_e * port_exp - bench_exp) * size_ret.loc[port_exp.index]).sum() / years)

    return {
        "corr_size_logcap": float(np.corrcoef(loading, np.log(cap0))[0, 1]),
        "held_to_unheld_cap": float(cap0[held].mean() / cap0[~held].mean()),
        "portfolio_size_loading": float(port_exp.mean()),
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
        "Ground-truth loadings used to generate returns. Not an input contract; "
        "provided to check the factor view."
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
