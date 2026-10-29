"""The published output of an engine run: :class:`Result`.

``Result`` wraps the contract-level :class:`attribution.result.AttributionResult`
(unchanged, schema 1.0.0) and adds what a reviewer needs to trust it:

* ``gross_to_net``   gross (holdings x prices) vs net (reported) returns and
                     the reconciling cost line, as named fields
* ``reconciliation`` every number behind the decision-vs-factor tie-out
* ``checks``         the input evidence the validation gate inspects
* ``limitations``    consolidated, structured: id, severity, text, affects
* ``run``            how the result was produced (versions, method, input hashes)

These records hold numbers as given; they do not re-validate themselves.
Judging them is the job of :mod:`attribution.validate`, the gate that refuses
to publish a broken result.
"""

from __future__ import annotations

import json
import typing
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal

from attribution.result import AttributionResult, ResultValidationError, _decode, _encode

__all__ = [
    "OUTPUT_SCHEMA_VERSION",
    "SEVERITIES",
    "Limitation",
    "GrossToNet",
    "ReconciliationDetail",
    "InputChecks",
    "RunInfo",
    "Result",
]

OUTPUT_SCHEMA_VERSION = "1.0.0"
Severity = Literal["info", "caution", "material"]
SEVERITIES: tuple[str, ...] = ("info", "caution", "material")


@dataclass(frozen=True)
class Limitation:
    """One limitation of the result. ``affects`` names the output sections it bears on."""

    id: str
    severity: Severity
    text: str
    affects: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.severity not in SEVERITIES:
            raise ResultValidationError(f"limitation {self.id!r}: severity must be one of {SEVERITIES}")
        if not self.id.strip() or not self.text.strip():
            raise ResultValidationError("limitation id and text must be non-empty")


@dataclass(frozen=True)
class GrossToNet:
    """Holdings-based (gross) versus reported (net) portfolio return.

    ``cost_*`` is net minus gross: negative means the reported return is lower.
    Monthly gap statistics come from the data-consistency check; when the mean
    absolute gap exceeds ``threshold_monthly`` the two inputs may not describe
    the same portfolio and ``consistent`` is False.
    """

    gross_cumulative: float
    net_cumulative: float
    cost_cumulative: float
    gross_annualised: float
    net_annualised: float
    cost_annualised: float
    mean_monthly_gap: float
    mean_abs_monthly_gap: float
    max_abs_monthly_gap: float
    threshold_monthly: float
    consistent: bool
    label: str


@dataclass(frozen=True)
class ReconciliationDetail:
    """Every number behind the tie-out.

    Which return series each view is built on:

    * decision layers (macro, allocation, selection, interaction): the GROSS
      portfolio return (holdings x prices, blended with cash at the equity
      weight) against the policy benchmark, linked with factors computed on
      that gross pair - so they sum to GROSS active;
    * cost line: NET minus GROSS cumulative portfolio return - so gross
      active + cost = NET active;
    * factor view: the NET portfolio return against the policy benchmark -
      so it sums to NET active.

    ``passes`` is True only if every difference is within ``tolerance``.
    """

    active_definition: str
    series_used: Mapping[str, str]
    linking: str
    active_return: float
    gross_active: float
    decision_layers_total: float
    cost_line: float
    decision_total: float
    factor_total: float
    layers_minus_gross_active: float
    gross_plus_cost_minus_net: float
    decision_minus_active: float
    factor_minus_active: float
    decision_minus_factor: float
    tolerance: float
    passes: bool
    decision_components: Mapping[str, float]
    factor_components: Mapping[str, float]
    max_period_identity_gap: float


@dataclass(frozen=True)
class InputChecks:
    """Evidence about the inputs, recorded so the gate (and a reviewer) can inspect it."""

    n_months: int
    holdings_weight_max_deviation: float
    benchmark_weight_max_deviation: float
    held_not_in_benchmark: tuple[str, ...]
    benchmark_not_held: tuple[str, ...]
    orphans_recorded: tuple[str, ...]
    dropped_holdings: tuple[str, ...]
    policy_equity_weight: float
    policy_equity_band: tuple[float, float]


@dataclass(frozen=True)
class RunInfo:
    package_version: str
    linking: str
    brinson_variant: str
    factor_order: tuple[str, ...]
    data_dir: str
    inputs: Mapping[str, str]
    generated_at_utc: datetime


@dataclass(frozen=True)
class Result:
    attribution: AttributionResult
    gross_to_net: GrossToNet
    reconciliation: ReconciliationDetail
    checks: InputChecks
    limitations: tuple[Limitation, ...]
    run: RunInfo
    output_schema_version: str = OUTPUT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return typing.cast(dict[str, Any], _encode(self))

    def to_json(self, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, allow_nan=False)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Result:
        return typing.cast(Result, _decode(cls, data, "result"))

    @classmethod
    def from_json(cls, text: str) -> Result:
        def reject_constant(token: str) -> float:
            raise ResultValidationError(f"JSON contains non-finite number {token}")

        return cls.from_dict(json.loads(text, parse_constant=reject_constant))

    def limitations_by_severity(self, severity: str) -> tuple[Limitation, ...]:
        return tuple(x for x in self.limitations if x.severity == severity)
