"""Output contract: the attribution result the app consumes.

:class:`AttributionResult` is an immutable tree of dataclasses that
serialises to JSON and back without loss. Construction validates internal
bookkeeping so a malformed result cannot be created, let alone shipped:

* ``totals.active == totals.portfolio - totals.benchmark`` (net), and
  ``totals.active_gross == totals.portfolio_gross - totals.benchmark`` (gross)
* decision layers are exactly macro, allocation, selection, interaction; they
  explain GROSS active return. ``decision_view.cost`` (fees and trading costs)
  bridges gross active to net active: ``net_total = total + cost``
* per-sector allocation/selection/interaction sum to the matching layers
* factors are exactly the six style factors, in contract order
* ``reconciliation.decision_total`` equals the decision layers plus the cost
  line (net); ``reconciliation.layers_total`` equals the four layers (gross)
* ``reconciliation.factor_total`` equals factor contributions + specific return
* ``reconciliation.difference`` and ``passes`` are what the numbers imply

All of the above hold to :data:`BOOKKEEPING_TOL`. A result whose two views do
*not* reconcile can still be built (``passes=False``) so the app can show the
break; the README states that a passing reconciliation is required of every
engine run.
"""

from __future__ import annotations

import dataclasses
import json
import math
import types
import typing
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from attribution.schemas import FACTORS, SECTORS

__all__ = [
    "SCHEMA_VERSION",
    "BOOKKEEPING_TOL",
    "DEFAULT_RECONCILIATION_TOL",
    "DECISION_LAYERS",
    "ResultValidationError",
    "Meta",
    "Period",
    "ConfidenceInterval",
    "Totals",
    "DecisionLayer",
    "SectorAttribution",
    "MacroAnatomy",
    "DecisionView",
    "FactorContribution",
    "SpecificReturn",
    "FactorDiagnostics",
    "FactorView",
    "Reconciliation",
    "AttributionResult",
]

#: 1.1.0: 13-sector taxonomy (Bankex and Energy retired); gross/net split in
#: totals, decision_view.cost, reconciliation.layers_total / gross_active.
SCHEMA_VERSION = "1.1.0"

#: Tolerance for sums that must hold by construction (floating-point noise only).
BOOKKEEPING_TOL = 1e-9

#: Default tolerance for decision view vs factor view: 0.1 basis point.
DEFAULT_RECONCILIATION_TOL = 1e-5

DECISION_LAYERS: tuple[str, ...] = ("macro", "allocation", "selection", "interaction")

Frequency = Literal["daily", "weekly", "monthly"]
Linking = Literal["arithmetic", "carino", "menchero", "geometric"]


class ResultValidationError(ValueError):
    """Raised when an AttributionResult (or part of one) breaks its contract."""


def _finite(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ResultValidationError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ResultValidationError(f"{name} must be finite, got {value!r}")


def _optional_finite(name: str, value: float | None) -> None:
    if value is not None:
        _finite(name, value)


def _close(a: float, b: float, tol: float = BOOKKEEPING_TOL) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Leaf records
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Meta:
    portfolio_id: str
    portfolio_name: str
    benchmark_name: str
    currency: str
    frequency: Frequency
    linking: Linking
    engine_version: str
    generated_at: datetime
    synthetic_data: bool
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("portfolio_id", "portfolio_name", "benchmark_name", "currency", "engine_version"):
            if not getattr(self, name).strip():
                raise ResultValidationError(f"meta.{name} must be non-empty")
        if self.generated_at.tzinfo is not None:
            raise ResultValidationError("meta.generated_at must be tz-naive")
        if self.schema_version != SCHEMA_VERSION:
            raise ResultValidationError(
                f"meta.schema_version {self.schema_version!r} does not match this package ({SCHEMA_VERSION!r})"
            )


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    n_periods: int

    def __post_init__(self) -> None:
        if self.end < self.start:
            raise ResultValidationError(f"period.end {self.end} is before period.start {self.start}")
        if self.n_periods < 1:
            raise ResultValidationError("period.n_periods must be at least 1")


@dataclass(frozen=True)
class ConfidenceInterval:
    """Interval around the active return (e.g. from a block bootstrap)."""

    level: float
    lower: float
    upper: float
    method: str

    def __post_init__(self) -> None:
        for name in ("level", "lower", "upper"):
            _finite(f"ci.{name}", getattr(self, name))
        if not 0.0 < self.level < 1.0:
            raise ResultValidationError(f"ci.level must be in (0, 1), got {self.level}")
        if self.lower > self.upper:
            raise ResultValidationError(f"ci.lower {self.lower} exceeds ci.upper {self.upper}")
        if not self.method.strip():
            raise ResultValidationError("ci.method must be non-empty")


@dataclass(frozen=True)
class Totals:
    """Cumulative returns over the period, simple decimals.

    ``portfolio`` / ``active`` are NET of fees (reported returns).
    ``portfolio_gross`` / ``active_gross`` are from holdings x prices, before
    fees and trading costs; both are optional and must be given together.
    """

    portfolio: float
    benchmark: float
    active: float
    ci: ConfidenceInterval
    portfolio_gross: float | None = None
    active_gross: float | None = None

    def __post_init__(self) -> None:
        for name in ("portfolio", "benchmark", "active"):
            _finite(f"totals.{name}", getattr(self, name))
        if not _close(self.active, self.portfolio - self.benchmark):
            raise ResultValidationError(
                f"totals.active {self.active} != portfolio - benchmark ({self.portfolio - self.benchmark})"
            )
        if (self.portfolio_gross is None) != (self.active_gross is None):
            raise ResultValidationError("totals.portfolio_gross and totals.active_gross must be given together")
        if self.portfolio_gross is not None and self.active_gross is not None:
            _finite("totals.portfolio_gross", self.portfolio_gross)
            _finite("totals.active_gross", self.active_gross)
            if not _close(self.active_gross, self.portfolio_gross - self.benchmark):
                raise ResultValidationError(
                    f"totals.active_gross {self.active_gross} != portfolio_gross - benchmark "
                    f"({self.portfolio_gross - self.benchmark})"
                )


@dataclass(frozen=True)
class DecisionLayer:
    name: str
    value: float

    def __post_init__(self) -> None:
        if self.name not in DECISION_LAYERS:
            raise ResultValidationError(f"unknown decision layer {self.name!r}")
        _finite(f"decision layer {self.name}", self.value)


@dataclass(frozen=True)
class SectorAttribution:
    sector: str
    portfolio_weight: float
    benchmark_weight: float
    portfolio_return: float
    benchmark_return: float
    allocation: float
    selection: float
    interaction: float

    def __post_init__(self) -> None:
        if self.sector not in SECTORS:
            raise ResultValidationError(f"by_sector: unknown sector {self.sector!r}")
        for f in dataclasses.fields(self):
            if f.name != "sector":
                _finite(f"by_sector[{self.sector}].{f.name}", getattr(self, f.name))


@dataclass(frozen=True)
class MacroAnatomy:
    """How the macro/exposure layer arises: equity exposure vs policy, and what cash earned."""

    average_equity_exposure: float
    policy_equity_exposure: float
    cash_return: float
    components: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _finite("macro_anatomy.average_equity_exposure", self.average_equity_exposure)
        _finite("macro_anatomy.policy_equity_exposure", self.policy_equity_exposure)
        _finite("macro_anatomy.cash_return", self.cash_return)
        for key, value in self.components.items():
            if not isinstance(key, str) or not key.strip():
                raise ResultValidationError("macro_anatomy.components keys must be non-empty strings")
            _finite(f"macro_anatomy.components[{key}]", value)


@dataclass(frozen=True)
class DecisionView:
    """Four decision layers explaining GROSS active return, plus the cost line.

    ``cost`` is fees and trading costs, expressed in the same linked
    (cumulative) space: ``total`` (the four layers) is gross active and
    ``net_total = total + cost`` is net active. The macro layer holds only
    the exposure-timing effect, never costs.
    """

    layers: tuple[DecisionLayer, ...]
    by_sector: tuple[SectorAttribution, ...]
    macro_anatomy: MacroAnatomy
    cost: float = 0.0

    def __post_init__(self) -> None:
        _finite("decision_view.cost", self.cost)
        names = tuple(layer.name for layer in self.layers)
        if names != DECISION_LAYERS:
            raise ResultValidationError(f"decision_view.layers must be {DECISION_LAYERS} in order, got {names}")
        sectors = [s.sector for s in self.by_sector]
        if len(set(sectors)) != len(sectors):
            raise ResultValidationError("decision_view.by_sector has duplicate sectors")
        components = self.macro_anatomy.components
        if components and not _close(sum(components.values()), self.layer("macro")):
            raise ResultValidationError(
                f"macro_anatomy.components sum to {sum(components.values())}, macro layer is {self.layer('macro')}"
            )
        if self.by_sector:
            for layer_name in ("allocation", "selection", "interaction"):
                total = sum(getattr(s, layer_name) for s in self.by_sector)
                if not _close(total, self.layer(layer_name)):
                    raise ResultValidationError(
                        f"by_sector {layer_name} sums to {total}, layer says {self.layer(layer_name)}"
                    )

    def layer(self, name: str) -> float:
        for layer in self.layers:
            if layer.name == name:
                return layer.value
        raise KeyError(name)

    @property
    def total(self) -> float:
        """The four layers: gross active return."""
        return sum(layer.value for layer in self.layers)

    @property
    def net_total(self) -> float:
        """Four layers plus the cost line: net active return."""
        return self.total + self.cost


@dataclass(frozen=True)
class FactorContribution:
    factor: str
    exposure: float
    factor_return: float
    contribution: float
    t_stat: float | None = None

    def __post_init__(self) -> None:
        if self.factor not in FACTORS:
            raise ResultValidationError(f"unknown factor {self.factor!r}")
        _finite(f"factor {self.factor}.exposure", self.exposure)
        _finite(f"factor {self.factor}.factor_return", self.factor_return)
        _finite(f"factor {self.factor}.contribution", self.contribution)
        _optional_finite(f"factor {self.factor}.t_stat", self.t_stat)


@dataclass(frozen=True)
class SpecificReturn:
    value: float
    t_stat: float | None = None
    information_ratio: float | None = None

    def __post_init__(self) -> None:
        _finite("specific_return.value", self.value)
        _optional_finite("specific_return.t_stat", self.t_stat)
        _optional_finite("specific_return.information_ratio", self.information_ratio)


@dataclass(frozen=True)
class FactorDiagnostics:
    method: str
    n_obs: int
    r_squared: float
    tracking_error: float
    residual_volatility: float
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.method.strip():
            raise ResultValidationError("diagnostics.method must be non-empty")
        if self.n_obs < 1:
            raise ResultValidationError("diagnostics.n_obs must be at least 1")
        for name in ("r_squared", "tracking_error", "residual_volatility"):
            _finite(f"diagnostics.{name}", getattr(self, name))
        if not 0.0 <= self.r_squared <= 1.0:
            raise ResultValidationError(f"diagnostics.r_squared must be in [0, 1], got {self.r_squared}")
        if self.tracking_error < 0 or self.residual_volatility < 0:
            raise ResultValidationError("diagnostics volatilities must be non-negative")


@dataclass(frozen=True)
class FactorView:
    factors: tuple[FactorContribution, ...]
    specific_return: SpecificReturn
    diagnostics: FactorDiagnostics

    def __post_init__(self) -> None:
        names = tuple(f.factor for f in self.factors)
        if names != FACTORS:
            raise ResultValidationError(f"factor_view.factors must be {FACTORS} in order, got {names}")

    @property
    def total(self) -> float:
        return sum(f.contribution for f in self.factors) + self.specific_return.value


@dataclass(frozen=True)
class Reconciliation:
    """Decision view vs factor view.

    ``decision_total`` is the decision view's NET total (four layers + cost);
    ``factor_total`` is the factor view's total; both explain net active.
    ``layers_total`` (the four layers) explains ``gross_active``; both are
    optional and must be given together.

    ``passes`` is true when both views agree with each other and with net
    active, and (if given) the layers agree with gross active, each within
    ``tolerance``.
    """

    decision_total: float
    factor_total: float
    difference: float
    tolerance: float
    passes: bool
    layers_total: float | None = None
    gross_active: float | None = None

    def __post_init__(self) -> None:
        for name in ("decision_total", "factor_total", "difference", "tolerance"):
            _finite(f"reconciliation.{name}", getattr(self, name))
        if self.tolerance < 0:
            raise ResultValidationError("reconciliation.tolerance must be non-negative")
        if not _close(self.difference, self.decision_total - self.factor_total):
            raise ResultValidationError(
                f"reconciliation.difference {self.difference} != decision_total - factor_total "
                f"({self.decision_total - self.factor_total})"
            )
        if (self.layers_total is None) != (self.gross_active is None):
            raise ResultValidationError("reconciliation.layers_total and gross_active must be given together")
        _optional_finite("reconciliation.layers_total", self.layers_total)
        _optional_finite("reconciliation.gross_active", self.gross_active)

    def expected_passes(self, active: float) -> bool:
        ok = abs(self.difference) <= self.tolerance and abs(self.decision_total - active) <= self.tolerance
        if self.layers_total is not None and self.gross_active is not None:
            ok = ok and abs(self.layers_total - self.gross_active) <= self.tolerance
        return ok

    @classmethod
    def compute(
        cls,
        decision_total: float,
        factor_total: float,
        active: float,
        tolerance: float = DEFAULT_RECONCILIATION_TOL,
        layers_total: float | None = None,
        gross_active: float | None = None,
    ) -> Reconciliation:
        difference = decision_total - factor_total
        draft = cls(decision_total, factor_total, difference, tolerance, False, layers_total, gross_active)
        return dataclasses.replace(draft, passes=draft.expected_passes(active))


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttributionResult:
    meta: Meta
    period: Period
    totals: Totals
    decision_view: DecisionView
    factor_view: FactorView
    reconciliation: Reconciliation
    limitations: tuple[str, ...]

    def __post_init__(self) -> None:
        rec = self.reconciliation
        if not _close(rec.decision_total, self.decision_view.net_total):
            raise ResultValidationError(
                f"reconciliation.decision_total {rec.decision_total} != decision layers + cost "
                f"{self.decision_view.net_total}"
            )
        if rec.layers_total is not None and not _close(rec.layers_total, self.decision_view.total):
            raise ResultValidationError(
                f"reconciliation.layers_total {rec.layers_total} != sum of decision layers {self.decision_view.total}"
            )
        if rec.gross_active is not None and self.totals.active_gross is not None and not _close(
            rec.gross_active, self.totals.active_gross
        ):
            raise ResultValidationError("reconciliation.gross_active != totals.active_gross")
        if not _close(rec.factor_total, self.factor_view.total):
            raise ResultValidationError(
                f"reconciliation.factor_total {rec.factor_total} != factor contributions + specific "
                f"{self.factor_view.total}"
            )
        expected = rec.expected_passes(self.totals.active)
        if rec.passes != expected:
            raise ResultValidationError(
                f"reconciliation.passes is {rec.passes} but the numbers imply {expected}"
            )
        for item in self.limitations:
            if not isinstance(item, str) or not item.strip():
                raise ResultValidationError("limitations must be non-empty strings")

    # --- serialisation -------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return typing.cast(dict[str, Any], _encode(self))

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> AttributionResult:
        return typing.cast(AttributionResult, _decode(cls, data, "result"))

    @classmethod
    def from_json(cls, text: str) -> AttributionResult:
        def reject_constant(token: str) -> float:
            raise ResultValidationError(f"JSON contains non-finite number {token}")

        return cls.from_dict(json.loads(text, parse_constant=reject_constant))


# ---------------------------------------------------------------------------
# Generic encode / decode driven by the dataclass type hints
# ---------------------------------------------------------------------------


def _encode(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {f.name: _encode(getattr(value, f.name)) for f in dataclasses.fields(value)}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, (tuple, list)):
        return [_encode(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _encode(v) for k, v in value.items()}
    if isinstance(value, float) and not math.isfinite(value):
        raise ResultValidationError(f"cannot serialise non-finite number {value}")
    return value


def _decode(tp: Any, data: Any, path: str) -> Any:
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)

    if origin in (typing.Union, types.UnionType):
        if data is None and type(None) in args:
            return None
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) != 1:
            raise TypeError(f"unsupported union at {path}: {tp}")
        return _decode(non_none[0], data, path)
    if origin is Literal:
        if data not in args:
            raise ResultValidationError(f"{path}: {data!r} is not one of {args}")
        return data
    if origin is tuple:
        if not isinstance(data, list):
            raise ResultValidationError(f"{path}: expected a list, got {type(data).__name__}")
        item_type = args[0]
        return tuple(_decode(item_type, v, f"{path}[{i}]") for i, v in enumerate(data))
    if origin in (dict, Mapping) or tp is Mapping:
        if not isinstance(data, dict):
            raise ResultValidationError(f"{path}: expected an object, got {type(data).__name__}")
        val_type = args[1] if args else Any
        return {str(k): _decode(val_type, v, f"{path}.{k}") for k, v in data.items()}
    if dataclasses.is_dataclass(tp):
        if not isinstance(data, dict):
            raise ResultValidationError(f"{path}: expected an object, got {type(data).__name__}")
        hints = typing.get_type_hints(tp)
        fields = {f.name: f for f in dataclasses.fields(tp)}
        unknown = sorted(set(data) - set(fields))
        if unknown:
            raise ResultValidationError(f"{path}: unexpected key(s) {unknown}")
        kwargs: dict[str, Any] = {}
        for name, f in fields.items():
            if name not in data:
                has_default = f.default is not dataclasses.MISSING or f.default_factory is not dataclasses.MISSING
                if has_default:
                    continue
                raise ResultValidationError(f"{path}: missing key {name!r}")
            kwargs[name] = _decode(hints[name], data[name], f"{path}.{name}")
        return tp(**kwargs)
    if tp is datetime:
        if not isinstance(data, str):
            raise ResultValidationError(f"{path}: expected an ISO datetime string")
        return datetime.fromisoformat(data)
    if tp is date:
        if not isinstance(data, str) or len(data) != 10:
            raise ResultValidationError(f"{path}: expected YYYY-MM-DD")
        return date.fromisoformat(data)
    if tp is bool:
        if not isinstance(data, bool):
            raise ResultValidationError(f"{path}: expected true/false, got {data!r}")
        return data
    if tp is int:
        if isinstance(data, bool) or not isinstance(data, int):
            raise ResultValidationError(f"{path}: expected an integer, got {data!r}")
        return data
    if tp is float:
        if isinstance(data, bool) or not isinstance(data, (int, float)):
            raise ResultValidationError(f"{path}: expected a number, got {data!r}")
        return float(data)
    if tp is str:
        if not isinstance(data, str):
            raise ResultValidationError(f"{path}: expected a string, got {data!r}")
        return data
    if tp is Any:
        return data
    raise TypeError(f"unsupported type at {path}: {tp}")
