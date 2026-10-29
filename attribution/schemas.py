"""Input contracts for the attribution engine.

Every table that enters the engine passes through a :class:`TableSchema`. The
schema checks column presence, types, value ranges and table-level rules, and
raises :class:`SchemaValidationError` naming the source, the column and the
offending rows. Nothing is silently coerced: a value that is not already what
the contract asks for is an error, not something to be fixed up.

Timing convention (applies to every dated table)
------------------------------------------------
A row dated ``t`` describes the period that *ends* on ``t``.

* ``Holdings`` and ``Benchmark`` weights dated ``t`` are the weights held
  through that period (i.e. set at the close of the previous period).
* Returns dated ``t`` are realised over the period ending ``t``.

So weights and returns join on the same date key with no shifting.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

import numpy as np
import pandas as pd

__all__ = [
    "Sector",
    "SECTORS",
    "INDEX_NAMES_NOT_SECTORS",
    "Factor",
    "FACTORS",
    "RegimeState",
    "REGIME_STATES",
    "CASH_KEY",
    "WEIGHT_SUM_TOL",
    "PERCENT_SCALE_MEDIAN",
    "Issue",
    "SchemaValidationError",
    "ColumnSpec",
    "TableSchema",
    "HOLDINGS",
    "SECURITY_MASTER",
    "PRICES",
    "BENCHMARK",
    "PORTFOLIO_RETURN",
    "FACTOR_RETURNS",
    "CASH_RETURN",
    "REGIME_SERIES",
    "TABLES",
    "Policy",
    "check_holdings_against_master",
    "isin_check_digit",
    "is_valid_isin",
]

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Sector(str, enum.Enum):
    """The fixed set of 13 sector names used throughout the engine.

    The list is a CONDENSED approximation of the NSE sector classification
    (the sector level of the NSE Indices industry classification), merged so
    that no two names overlap:

    * financials of every kind - banks, NBFCs, insurers, AMCs - are
      ``Financial Services`` (index names such as "Bankex" are not sectors);
    * oil, gas and consumable fuels are ``Oil & Gas`` (there is no separate
      "Energy", which would overlap Oil & Gas and Power);
    * chemicals, construction materials and similar are ``Basic Materials``.

    A security belongs to exactly one of these. Changing this list is a
    contract change: bump ``result.SCHEMA_VERSION`` and update the README.
    """

    AUTO = "Auto"
    BASIC_MATERIALS = "Basic Materials"
    CAPITAL_GOODS = "Capital Goods"
    CONSUMER_DURABLES = "Consumer Durables"
    FINANCIAL_SERVICES = "Financial Services"
    FMCG = "FMCG"
    HEALTHCARE = "Healthcare"
    INFORMATION_TECHNOLOGY = "Information Technology"
    METAL = "Metal"
    OIL_AND_GAS = "Oil & Gas"
    POWER = "Power"
    REALTY = "Realty"
    TELECOMMUNICATION = "Telecommunication"


SECTORS: tuple[str, ...] = tuple(s.value for s in Sector)
assert len(SECTORS) == 13
#: Index names that must never be used as sector names.
INDEX_NAMES_NOT_SECTORS: frozenset[str] = frozenset({"Bankex", "Bank Nifty", "Nifty Bank", "Sensex", "Nifty 50", "Nifty Financial Services"})
assert not INDEX_NAMES_NOT_SECTORS & set(SECTORS)


class Factor(str, enum.Enum):
    """Style factors for the factor view. Specific return is the residual."""

    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"


FACTORS: tuple[str, ...] = tuple(f.value for f in Factor)


class RegimeState(str, enum.Enum):
    """States emitted by the regime model (raw) and its confirmation filter."""

    NORMAL = "normal"
    ELEVATED = "elevated"
    CRISIS = "crisis"


REGIME_STATES: tuple[str, ...] = tuple(s.value for s in RegimeState)

#: Key used for the cash sleeve in :class:`Policy` weights and bands.
CASH_KEY = "Cash"

#: Absolute tolerance for "weights sum to 1.0 per date".
WEIGHT_SUM_TOL = 1e-6

#: If the median absolute return in a column exceeds this, the column is
#: rejected as probably being in percent rather than decimals. 0.2 (20%) is far
#: above any plausible median monthly or daily simple return.
PERCENT_SCALE_MEDIAN = 0.2

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

_MAX_LOCATIONS_SHOWN = 10


@dataclass(frozen=True)
class Issue:
    """One validation failure: where it is and what is wrong."""

    column: str | None
    locations: tuple[str, ...]
    message: str

    def render(self) -> str:
        parts: list[str] = []
        if self.column is not None:
            parts.append(f"column '{self.column}'")
        if self.locations:
            shown = ", ".join(self.locations[:_MAX_LOCATIONS_SHOWN])
            extra = len(self.locations) - _MAX_LOCATIONS_SHOWN
            if extra > 0:
                shown += f" (+{extra} more)"
            parts.append(shown)
        where = "; ".join(parts)
        return f"{where}: {self.message}" if where else self.message


class SchemaValidationError(ValueError):
    """Raised when input does not meet its contract. Lists every issue found."""

    def __init__(self, table: str, source: str, issues: Sequence[Issue]) -> None:
        self.table = table
        self.source = source
        self.issues: tuple[Issue, ...] = tuple(issues)
        lines = [f"{table} validation failed for {source} ({len(self.issues)} issue(s)):"]
        lines += [f"  - {issue.render()}" for issue in self.issues]
        super().__init__("\n".join(lines))


# ---------------------------------------------------------------------------
# ISIN
# ---------------------------------------------------------------------------

_ISIN_RE = re.compile(r"^IN[A-Z0-9]{9}[0-9]$")


def _isin_digits(body: str) -> str:
    return "".join(str(int(ch, 36)) for ch in body)


def isin_check_digit(first_eleven: str) -> int:
    """Return the ISO 6166 (Luhn) check digit for the first 11 ISIN characters."""
    if len(first_eleven) != 11 or not first_eleven.isalnum() or first_eleven != first_eleven.upper():
        raise ValueError(f"expected 11 upper-case alphanumeric characters, got {first_eleven!r}")
    digits = _isin_digits(first_eleven)
    total = 0
    # Luhn: double every second digit starting from the rightmost of the payload.
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 0:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return (10 - total % 10) % 10


def is_valid_isin(value: str) -> bool:
    """True for a well-formed Indian ISIN (``IN`` prefix) with a correct check digit."""
    if not isinstance(value, str) or not _ISIN_RE.match(value):
        return False
    return isin_check_digit(value[:11]) == int(value[11])


# ---------------------------------------------------------------------------
# Column and table specs
# ---------------------------------------------------------------------------

ColumnKind = Literal[
    "date",  # tz-naive calendar date, no time component
    "id",  # non-empty string without surrounding whitespace
    "text",  # non-empty string without surrounding whitespace
    "sector",  # one of SECTORS, exact match
    "isin",  # valid Indian ISIN
    "factor",  # one of FACTORS
    "regime_state",  # one of REGIME_STATES
    "return",  # simple decimal return, finite, > -1
    "weight",  # finite, within [0, 1]
    "fraction",  # finite, within [0, 1]
]

_NUMERIC_KINDS = frozenset({"return", "weight", "fraction"})
_ENUM_KINDS: dict[str, tuple[str, ...]] = {
    "sector": SECTORS,
    "factor": FACTORS,
    "regime_state": REGIME_STATES,
}


@dataclass(frozen=True)
class ColumnSpec:
    name: str
    kind: ColumnKind


RowLabeler = Callable[[int], str]
TableCheck = Callable[[pd.DataFrame, RowLabeler], list[Issue]]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value == ""
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _fmt_value(value: Any) -> str:
    text = repr(value)
    return text if len(text) <= 40 else text[:37] + "..."


def _collect(
    column: str,
    bad: list[tuple[int, Any]],
    label: RowLabeler,
    message: str,
    show_values: bool = True,
) -> Issue:
    locations = tuple(label(i) for i, _ in bad)
    if show_values:
        sample = ", ".join(_fmt_value(v) for _, v in bad[:5])
        message = f"{message} (got {sample}{', ...' if len(bad) > 5 else ''})"
    return Issue(column=column, locations=locations, message=message)


def _convert_date(values: Sequence[Any], column: str, label: RowLabeler) -> tuple[pd.Series | None, list[Issue]]:
    out: list[pd.Timestamp] = []
    missing: list[tuple[int, Any]] = []
    unparsable: list[tuple[int, Any]] = []
    tz_aware: list[tuple[int, Any]] = []
    has_time: list[tuple[int, Any]] = []
    for i, raw in enumerate(values):
        if _is_missing(raw):
            missing.append((i, raw))
            continue
        if isinstance(raw, (pd.Timestamp, datetime, date, np.datetime64)):
            ts = pd.Timestamp(raw)
        elif isinstance(raw, str):
            if raw != raw.strip():
                unparsable.append((i, raw))
                continue
            try:
                ts = pd.Timestamp(raw)
            except (ValueError, TypeError):
                unparsable.append((i, raw))
                continue
        else:
            unparsable.append((i, raw))
            continue
        if ts.tzinfo is not None:
            tz_aware.append((i, raw))
            continue
        if ts != ts.normalize():
            has_time.append((i, raw))
            continue
        out.append(ts)
    issues: list[Issue] = []
    if missing:
        issues.append(_collect(column, missing, label, "missing date", show_values=False))
    if unparsable:
        issues.append(_collect(column, unparsable, label, "not an ISO calendar date (YYYY-MM-DD)"))
    if tz_aware:
        issues.append(_collect(column, tz_aware, label, "dates must be tz-naive; found a timezone"))
    if has_time:
        issues.append(_collect(column, has_time, label, "dates must not carry a time of day"))
    if issues:
        return None, issues
    return pd.Series(pd.DatetimeIndex(out).as_unit("ns"), name=column), []


def _convert_string(
    values: Sequence[Any], column: str, label: RowLabeler, kind: str
) -> tuple[pd.Series | None, list[Issue]]:
    missing: list[tuple[int, Any]] = []
    not_str: list[tuple[int, Any]] = []
    padded: list[tuple[int, Any]] = []
    for i, raw in enumerate(values):
        if _is_missing(raw):
            missing.append((i, raw))
        elif not isinstance(raw, str):
            not_str.append((i, raw))
        elif raw != raw.strip():
            padded.append((i, raw))
    issues: list[Issue] = []
    if missing:
        issues.append(_collect(column, missing, label, "missing value", show_values=False))
    if not_str:
        issues.append(_collect(column, not_str, label, "expected text"))
    if padded:
        issues.append(_collect(column, padded, label, "leading or trailing whitespace"))
    if issues:
        return None, issues

    strings = [str(v) for v in values]
    if kind in _ENUM_KINDS:
        allowed = _ENUM_KINDS[kind]
        unknown = [(i, v) for i, v in enumerate(strings) if v not in allowed]
        if unknown:
            issues.append(
                _collect(column, unknown, label, f"not an allowed {kind} value; allowed: {', '.join(allowed)}")
            )
    elif kind == "isin":
        bad = [(i, v) for i, v in enumerate(strings) if not is_valid_isin(v)]
        if bad:
            issues.append(
                _collect(column, bad, label, "invalid ISIN (expect IN + 9 alphanumerics + check digit)")
            )
    if issues:
        return None, issues
    return pd.Series(strings, name=column, dtype=object), []


def _convert_numeric(
    values: Sequence[Any], column: str, label: RowLabeler, kind: str
) -> tuple[pd.Series | None, list[Issue]]:
    numbers: list[float] = []
    missing: list[tuple[int, Any]] = []
    non_numeric: list[tuple[int, Any]] = []
    non_finite: list[tuple[int, Any]] = []
    for i, raw in enumerate(values):
        if _is_missing(raw):
            missing.append((i, raw))
            continue
        if isinstance(raw, (bool, np.bool_)):
            non_numeric.append((i, raw))
            continue
        if isinstance(raw, (int, float, np.integer, np.floating)):
            x = float(raw)
        elif isinstance(raw, str):
            try:
                x = float(raw)
            except ValueError:
                non_numeric.append((i, raw))
                continue
        else:
            non_numeric.append((i, raw))
            continue
        if not math.isfinite(x):
            non_finite.append((i, raw))
            continue
        numbers.append(x)
    issues: list[Issue] = []
    if missing:
        issues.append(_collect(column, missing, label, "missing value", show_values=False))
    if non_numeric:
        issues.append(_collect(column, non_numeric, label, "not a plain decimal number (no '%' or text)"))
    if non_finite:
        issues.append(_collect(column, non_finite, label, "must be finite"))
    if issues:
        return None, issues

    arr = np.asarray(numbers, dtype=float)
    if kind == "return":
        low = [(i, v) for i, v in enumerate(arr.tolist()) if v <= -1.0]
        if low:
            issues.append(_collect(column, low, label, "simple return must be greater than -1"))
        if arr.size and float(np.median(np.abs(arr))) > PERCENT_SCALE_MEDIAN:
            issues.append(
                Issue(
                    column=column,
                    locations=(),
                    message=(
                        f"median absolute return is {float(np.median(np.abs(arr))):.4g}, "
                        f"above {PERCENT_SCALE_MEDIAN}; values look like percent. "
                        "Supply simple decimal returns (0.012 for 1.2%)"
                    ),
                )
            )
    else:  # weight, fraction
        out_of_range = [(i, v) for i, v in enumerate(arr.tolist()) if v < 0.0 or v > 1.0]
        if out_of_range:
            issues.append(_collect(column, out_of_range, label, f"{kind} must be within [0, 1]"))
    if issues:
        return None, issues
    return pd.Series(arr, name=column), []


@dataclass(frozen=True)
class TableSchema:
    """Contract for one input table."""

    name: str
    columns: tuple[ColumnSpec, ...]
    key: tuple[str, ...]
    checks: tuple[TableCheck, ...] = field(default=())

    @property
    def column_names(self) -> tuple[str, ...]:
        return tuple(c.name for c in self.columns)

    def validate(
        self,
        df: pd.DataFrame,
        source: str = "<dataframe>",
        row_labels: Sequence[str] | None = None,
    ) -> pd.DataFrame:
        """Validate ``df`` and return a typed copy in contract column order.

        ``row_labels`` gives a human label per row (for example ``"line 7"``
        for a CSV). By default rows are labelled by their DataFrame index.
        Raises :class:`SchemaValidationError` listing every problem found.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"{self.name}: expected a pandas DataFrame, got {type(df).__name__}")
        if row_labels is not None and len(row_labels) != len(df):
            raise ValueError("row_labels must have one entry per row")

        index_values = list(df.index)
        labels: list[str] = (
            list(row_labels) if row_labels is not None else [f"index {v}" for v in index_values]
        )

        def label(i: int) -> str:
            return labels[i]

        present = [str(c) for c in df.columns]
        missing = [c for c in self.column_names if c not in present]
        unexpected = [c for c in present if c not in self.column_names]
        duplicated = sorted({c for c in present if present.count(c) > 1})
        structural: list[Issue] = []
        if missing:
            structural.append(Issue(None, (), f"missing column(s): {', '.join(missing)}"))
        if unexpected:
            structural.append(
                Issue(None, (), f"unexpected column(s): {', '.join(unexpected)}; expected exactly: "
                      f"{', '.join(self.column_names)}")
            )
        if duplicated:
            structural.append(Issue(None, (), f"duplicated column(s): {', '.join(duplicated)}"))
        if structural:
            raise SchemaValidationError(self.name, source, structural)
        if len(df) == 0:
            raise SchemaValidationError(self.name, source, [Issue(None, (), "table has no rows")])

        issues: list[Issue] = []
        converted: dict[str, pd.Series] = {}
        for spec in self.columns:
            values = df[spec.name].tolist()
            if spec.kind == "date":
                series, col_issues = _convert_date(values, spec.name, label)
            elif spec.kind in _NUMERIC_KINDS:
                series, col_issues = _convert_numeric(values, spec.name, label, spec.kind)
            else:
                series, col_issues = _convert_string(values, spec.name, label, spec.kind)
            issues.extend(col_issues)
            if series is not None:
                converted[spec.name] = series.reset_index(drop=True)
        if issues:
            raise SchemaValidationError(self.name, source, issues)

        out = pd.DataFrame({name: converted[name] for name in self.column_names})

        dup_mask = out.duplicated(subset=list(self.key), keep=False)
        if bool(dup_mask.any()):
            positions = [int(i) for i in np.flatnonzero(dup_mask.to_numpy())]
            issues.append(
                Issue(
                    column=", ".join(self.key),
                    locations=tuple(label(i) for i in positions),
                    message=f"duplicate key; each ({', '.join(self.key)}) must appear once",
                )
            )
        for check in self.checks:
            issues.extend(check(out, label))
        if issues:
            raise SchemaValidationError(self.name, source, issues)
        return out


def _weights_sum_to_one(df: pd.DataFrame, label: RowLabeler) -> list[Issue]:
    sums = df.groupby("date", sort=True)["weight"].sum()
    bad = sums[(sums - 1.0).abs() > WEIGHT_SUM_TOL]
    if bad.empty:
        return []
    issues: list[Issue] = []
    for d in bad.index:
        rows = tuple(label(int(i)) for i in np.flatnonzero((df["date"] == d).to_numpy()))
        issues.append(
            Issue(
                column="weight",
                locations=rows,
                message=f"weights for {d.date().isoformat()} sum to {bad[d]:.9f}, "
                f"must be 1.0 within {WEIGHT_SUM_TOL:g}",
            )
        )
    return issues


# ---------------------------------------------------------------------------
# The contracts
# ---------------------------------------------------------------------------

HOLDINGS = TableSchema(
    name="Holdings",
    columns=(
        ColumnSpec("date", "date"),
        ColumnSpec("security_id", "id"),
        ColumnSpec("weight", "weight"),
        ColumnSpec("sector", "sector"),
    ),
    key=("date", "security_id"),
    checks=(_weights_sum_to_one,),
)

SECURITY_MASTER = TableSchema(
    name="SecurityMaster",
    columns=(
        ColumnSpec("security_id", "id"),
        ColumnSpec("name", "text"),
        ColumnSpec("sector", "sector"),
        ColumnSpec("isin", "isin"),
    ),
    key=("security_id",),
)

PRICES = TableSchema(
    name="Prices",
    columns=(
        ColumnSpec("date", "date"),
        ColumnSpec("security_id", "id"),
        ColumnSpec("total_return", "return"),
    ),
    key=("date", "security_id"),
)

BENCHMARK = TableSchema(
    name="Benchmark",
    columns=(
        ColumnSpec("date", "date"),
        ColumnSpec("sector", "sector"),
        ColumnSpec("weight", "weight"),
        ColumnSpec("total_return", "return"),
    ),
    key=("date", "sector"),
    checks=(_weights_sum_to_one,),
)

PORTFOLIO_RETURN = TableSchema(
    name="PortfolioReturn",
    columns=(ColumnSpec("date", "date"), ColumnSpec("total_return", "return")),
    key=("date",),
)

FACTOR_RETURNS = TableSchema(
    name="FactorReturns",
    columns=(
        ColumnSpec("date", "date"),
        ColumnSpec("factor", "factor"),
        ColumnSpec("total_return", "return"),
    ),
    key=("date", "factor"),
)

CASH_RETURN = TableSchema(
    name="CashReturn",
    columns=(ColumnSpec("date", "date"), ColumnSpec("total_return", "return")),
    key=("date",),
)

REGIME_SERIES = TableSchema(
    name="RegimeSeries",
    columns=(
        ColumnSpec("date", "date"),
        ColumnSpec("raw_state", "regime_state"),
        ColumnSpec("confirmed_state", "regime_state"),
        ColumnSpec("equity_exposure", "fraction"),
    ),
    key=("date",),
)

TABLES: Mapping[str, TableSchema] = {
    s.name: s
    for s in (
        HOLDINGS,
        SECURITY_MASTER,
        PRICES,
        BENCHMARK,
        PORTFOLIO_RETURN,
        FACTOR_RETURNS,
        CASH_RETURN,
        REGIME_SERIES,
    )
}


def check_holdings_against_master(
    holdings: pd.DataFrame, master: pd.DataFrame, source: str = "Holdings vs SecurityMaster"
) -> None:
    """Cross-check validated holdings against a validated security master.

    Every held security must exist in the master, and the sector on each
    holding row must match the master's sector for that security.
    """
    sector_of = dict(zip(master["security_id"], master["sector"], strict=True))
    unknown = sorted({s for s in holdings["security_id"] if s not in sector_of})
    issues: list[Issue] = []
    if unknown:
        issues.append(Issue("security_id", tuple(unknown), "security_id not in SecurityMaster"))
    mismatched = [
        f"{row.security_id} on {row.date.date().isoformat()} ({row.sector} vs master {sector_of[row.security_id]})"
        for row in holdings.itertuples(index=False)
        if row.security_id in sector_of and row.sector != sector_of[row.security_id]
    ]
    if mismatched:
        issues.append(Issue("sector", tuple(mismatched), "holding sector differs from SecurityMaster"))
    if issues:
        raise SchemaValidationError("Holdings", source, issues)


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------

DeriskMode = Literal["cash", "within_equity"]
_DERISK_MODES: tuple[str, ...] = ("cash", "within_equity")
_POLICY_KEYS = frozenset({"neutral_weights", "bands", "derisk_mode", "review_date", "benchmark_name"})


@dataclass(frozen=True)
class Policy:
    """Strategic policy the decision view measures against.

    ``neutral_weights`` maps each of the 13 sectors, plus optionally ``"Cash"``,
    to a neutral weight of the *total* portfolio; together they sum to 1.
    ``bands`` gives the permitted (low, high) range for the same keys, with
    ``0 <= low <= neutral <= high <= 1``.

    ``derisk_mode`` says how the manager reduces risk: ``"cash"`` moves money
    out of equity into cash; ``"within_equity"`` stays fully invested and
    rotates between sectors.
    """

    neutral_weights: Mapping[str, float]
    bands: Mapping[str, tuple[float, float]]
    derisk_mode: DeriskMode
    review_date: date
    benchmark_name: str

    def __post_init__(self) -> None:
        problems: list[str] = []
        allowed = set(SECTORS) | {CASH_KEY}

        if not isinstance(self.neutral_weights, Mapping) or not self.neutral_weights:
            problems.append("neutral_weights must be a non-empty mapping")
        if not isinstance(self.bands, Mapping):
            problems.append("bands must be a mapping")
        if problems:
            raise SchemaValidationError("Policy", "<policy>", [Issue(None, (), p) for p in problems])

        nw_keys = set(self.neutral_weights)
        unknown = sorted(nw_keys - allowed)
        if unknown:
            problems.append(f"neutral_weights has unknown key(s): {', '.join(unknown)}")
        missing_sectors = sorted(set(SECTORS) - nw_keys)
        if missing_sectors:
            problems.append(f"neutral_weights missing sector(s): {', '.join(missing_sectors)}")
        if set(self.bands) != nw_keys:
            only_bands = sorted(set(self.bands) - nw_keys)
            only_nw = sorted(nw_keys - set(self.bands))
            if only_bands:
                problems.append(f"bands has key(s) not in neutral_weights: {', '.join(only_bands)}")
            if only_nw:
                problems.append(f"bands missing key(s): {', '.join(only_nw)}")

        for key, w in self.neutral_weights.items():
            if isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(float(w)):
                problems.append(f"neutral_weights[{key!r}] must be a finite number, got {w!r}")
            elif not 0.0 <= float(w) <= 1.0:
                problems.append(f"neutral_weights[{key!r}] must be within [0, 1], got {w}")
        if not problems:
            total = sum(float(w) for w in self.neutral_weights.values())
            if abs(total - 1.0) > WEIGHT_SUM_TOL:
                problems.append(f"neutral_weights sum to {total:.9f}, must be 1.0 within {WEIGHT_SUM_TOL:g}")

        for key, band in self.bands.items():
            if not isinstance(band, tuple) or len(band) != 2:
                problems.append(f"bands[{key!r}] must be a (low, high) tuple, got {band!r}")
                continue
            lo, hi = band
            if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x)) for x in band):
                problems.append(f"bands[{key!r}] must hold finite numbers, got {band!r}")
                continue
            if not 0.0 <= lo <= hi <= 1.0:
                problems.append(f"bands[{key!r}] must satisfy 0 <= low <= high <= 1, got {band!r}")
            nw = self.neutral_weights.get(key)
            if isinstance(nw, (int, float)) and not isinstance(nw, bool) and not lo <= nw <= hi:
                problems.append(f"neutral weight {nw} for {key!r} lies outside its band {band!r}")

        if self.derisk_mode not in _DERISK_MODES:
            problems.append(f"derisk_mode must be one of {', '.join(_DERISK_MODES)}, got {self.derisk_mode!r}")
        if isinstance(self.review_date, datetime) or not isinstance(self.review_date, date):
            problems.append(f"review_date must be a datetime.date (no time, no timezone), got {self.review_date!r}")
        if not isinstance(self.benchmark_name, str) or not self.benchmark_name.strip():
            problems.append("benchmark_name must be a non-empty string")
        elif self.benchmark_name != self.benchmark_name.strip():
            problems.append("benchmark_name has leading or trailing whitespace")
        if self.derisk_mode == "cash" and CASH_KEY not in self.neutral_weights:
            problems.append(f"derisk_mode 'cash' requires a {CASH_KEY!r} entry in neutral_weights and bands")

        if problems:
            raise SchemaValidationError("Policy", "<policy>", [Issue(None, (), p) for p in problems])

    @property
    def neutral_equity_exposure(self) -> float:
        """Neutral weight of the equity sleeve: 1 minus the neutral cash weight."""
        return 1.0 - float(self.neutral_weights.get(CASH_KEY, 0.0))

    def to_dict(self) -> dict[str, Any]:
        return {
            "neutral_weights": {k: float(v) for k, v in self.neutral_weights.items()},
            "bands": {k: [float(lo), float(hi)] for k, (lo, hi) in self.bands.items()},
            "derisk_mode": self.derisk_mode,
            "review_date": self.review_date.isoformat(),
            "benchmark_name": self.benchmark_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], source: str = "<policy>") -> Policy:
        """Build a Policy from plain JSON-style data, failing on anything unexpected."""
        if not isinstance(data, Mapping):
            raise SchemaValidationError("Policy", source, [Issue(None, (), "policy must be a JSON object")])
        problems: list[str] = []
        missing = sorted(_POLICY_KEYS - set(data))
        extra = sorted(set(data) - _POLICY_KEYS)
        if missing:
            problems.append(f"missing key(s): {', '.join(missing)}")
        if extra:
            problems.append(f"unexpected key(s): {', '.join(extra)}")
        if problems:
            raise SchemaValidationError("Policy", source, [Issue(None, (), p) for p in problems])

        if not isinstance(data["neutral_weights"], Mapping):
            problems.append("neutral_weights must be an object")
        bands_raw = data["bands"]
        bands: dict[str, tuple[float, float]] = {}
        if isinstance(bands_raw, Mapping):
            for k, v in bands_raw.items():
                if isinstance(v, (list, tuple)) and len(v) == 2:
                    bands[k] = (v[0], v[1])
                else:
                    problems.append(f"bands[{k!r}] must be a two-element list, got {v!r}")
        else:
            problems.append("bands must be an object")
        review_raw = data["review_date"]
        review: date | None = None
        if isinstance(review_raw, str):
            try:
                review = date.fromisoformat(review_raw)
            except ValueError:
                problems.append(f"review_date must be YYYY-MM-DD, got {review_raw!r}")
            else:
                if len(review_raw) != 10:
                    problems.append(f"review_date must be YYYY-MM-DD, got {review_raw!r}")
        else:
            problems.append(f"review_date must be a YYYY-MM-DD string, got {review_raw!r}")
        if problems:
            raise SchemaValidationError("Policy", source, [Issue(None, (), p) for p in problems])

        try:
            return cls(
                neutral_weights=dict(data["neutral_weights"]),
                bands=bands,
                derisk_mode=data["derisk_mode"],
                review_date=review,  # type: ignore[arg-type]
                benchmark_name=data["benchmark_name"],
            )
        except SchemaValidationError as exc:
            raise SchemaValidationError("Policy", source, exc.issues) from None
