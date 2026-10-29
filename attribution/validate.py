"""The gate: refuse to publish a broken result.

:func:`validate_result` inspects a finished :class:`attribution.outputs.Result`
and raises :class:`GateError` naming every failure it finds:

1. a reconciliation identity breaks beyond tolerance - checked separately:
   the four decision layers == GROSS active; gross active + cost line ==
   NET active; factor view total == NET active
2. portfolio or benchmark weights do not sum to 1.0 within 1e-6 in a period
3. a NaN or infinity anywhere in the output figures
4. fewer than 24 months
5. the policy equity weight lies outside its own stated band
6. a sector held but not in the benchmark, or the reverse, that is missing
   from the recorded orphan-sector list

A result that fails is never written; callers keep the last good file.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from typing import Any

from attribution.outputs import Result

__all__ = ["MIN_MONTHS", "WEIGHT_TOL", "GateError", "check_min_months", "find_non_finite", "validate_result"]

MIN_MONTHS = 24
WEIGHT_TOL = 1e-6
PERIOD_IDENTITY_TOL = 1e-9


class GateError(ValueError):
    """The result failed validation and must not be published."""

    def __init__(self, failures: list[str]) -> None:
        self.failures = tuple(failures)
        lines = [f"VALIDATION GATE FAILED - result not published ({len(failures)} problem(s)):"]
        lines += [f"  - {f}" for f in failures]
        super().__init__("\n".join(lines))


def check_min_months(n_months: int) -> None:
    """Raise straight away if the window is too short to attribute."""
    if n_months < MIN_MONTHS:
        raise GateError([f"period has {n_months} months; at least {MIN_MONTHS} are required"])


def find_non_finite(obj: Any, path: str = "result") -> list[str]:
    """Paths of every NaN or infinite number inside a dataclass/mapping/sequence tree."""
    found: list[str] = []
    if isinstance(obj, bool) or obj is None:
        return found
    if isinstance(obj, float):
        if not math.isfinite(obj):
            found.append(f"{path} = {obj}")
        return found
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        for f in dataclasses.fields(obj):
            found += find_non_finite(getattr(obj, f.name), f"{path}.{f.name}")
    elif isinstance(obj, Mapping):
        for k, v in obj.items():
            found += find_non_finite(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            found += find_non_finite(v, f"{path}[{i}]")
    return found


def validate_result(result: Result) -> Result:
    """Return ``result`` unchanged if it passes every check; otherwise raise :class:`GateError`."""
    failures: list[str] = []

    # 3. non-finite numbers first: later checks would be meaningless on them
    bad = find_non_finite(result)
    if bad:
        shown = ", ".join(bad[:10]) + (f" (+{len(bad) - 10} more)" if len(bad) > 10 else "")
        failures.append(f"non-finite number(s) in the output: {shown}")

    # 1. reconciliation - recomputed from the recorded figures, not read from flags
    rec = result.reconciliation
    tol = rec.tolerance
    checks_rec = (
        ("the four decision layers vs GROSS active", rec.decision_layers_total, rec.gross_active),
        ("gross active + cost line vs NET active", rec.gross_active + rec.cost_line, rec.active_return),
        ("factor view total vs NET active", rec.factor_total, rec.active_return),
        ("decision view (layers + cost) vs factor view", rec.decision_layers_total + rec.cost_line, rec.factor_total),
    )
    for label, a_val, b_val in checks_rec:
        if not (math.isfinite(a_val) and math.isfinite(b_val)) or abs(a_val - b_val) > tol:
            failures.append(f"reconciliation broken: {label} differ by {a_val - b_val:.3e} (tolerance {tol:g}); "
                            f"values {a_val:.12f} and {b_val:.12f}")
    if math.isfinite(rec.decision_total) and abs(rec.decision_total - (rec.decision_layers_total + rec.cost_line)) > tol:
        failures.append("reconciliation broken: decision_total is not the four layers plus the cost line")
    if all(math.isfinite(v) for v in rec.decision_components.values()) and abs(
        sum(rec.decision_components.values()) - rec.decision_layers_total
    ) > tol:
        failures.append("reconciliation broken: the four decision components do not sum to the layers total")
    if set(rec.decision_components) != {"macro", "allocation", "selection", "interaction"}:
        failures.append(f"reconciliation: decision components must be exactly the four layers, got "
                        f"{sorted(rec.decision_components)}")
    if all(math.isfinite(v) for v in rec.factor_components.values()) and abs(
        sum(rec.factor_components.values()) - rec.factor_total
    ) > tol:
        failures.append("reconciliation broken: factor components do not sum to the factor total")
    if not rec.max_period_identity_gap <= PERIOD_IDENTITY_TOL:
        failures.append(f"reconciliation broken: a single-period identity is off by {rec.max_period_identity_gap:.3e} "
                        f"(tolerance {PERIOD_IDENTITY_TOL:g})")
    inner = result.attribution.reconciliation
    if not inner.passes:
        failures.append(f"reconciliation broken: attribution.reconciliation.passes is False "
                        f"(difference {inner.difference:.3e})")
    if not rec.passes:
        failures.append("reconciliation broken: reconciliation.passes is False")
    totals = result.attribution.totals
    if abs(totals.active - rec.active_return) > tol:
        failures.append("reconciliation broken: totals.active differs from the net active return used for reconciliation")
    if totals.active_gross is None or abs(totals.active_gross - rec.gross_active) > tol:
        failures.append("reconciliation broken: totals.active_gross missing or different from the gross active used")
    macro_parts = result.attribution.decision_view.macro_anatomy.components
    if set(macro_parts) != {"exposure"}:
        failures.append(f"macro layer must contain only exposure timing, found components {sorted(macro_parts)}")

    # 2. weights
    chk = result.checks
    if not chk.holdings_weight_max_deviation <= WEIGHT_TOL:
        failures.append(f"portfolio weights do not sum to 1.0 in some period (max deviation "
                        f"{chk.holdings_weight_max_deviation:.3e}, tolerance {WEIGHT_TOL:g})")
    if not chk.benchmark_weight_max_deviation <= WEIGHT_TOL:
        failures.append(f"benchmark weights do not sum to 1.0 in some period (max deviation "
                        f"{chk.benchmark_weight_max_deviation:.3e}, tolerance {WEIGHT_TOL:g})")

    # 4. length
    if chk.n_months < MIN_MONTHS:
        failures.append(f"period has {chk.n_months} months; at least {MIN_MONTHS} are required")
    if result.attribution.period.n_periods != chk.n_months:
        failures.append("period.n_periods does not match the number of months checked")

    # 5. policy band
    lo, hi = chk.policy_equity_band
    w = chk.policy_equity_weight
    if not (lo - 1e-12 <= w <= hi + 1e-12):
        failures.append(f"policy equity weight {w:.6f} lies outside its own band [{lo:.6f}, {hi:.6f}]")

    # 6. orphan sectors
    recorded = set(chk.orphans_recorded)
    missing = sorted((set(chk.held_not_in_benchmark) | set(chk.benchmark_not_held)) - recorded)
    if missing:
        shown = ", ".join(missing[:10]) + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
        failures.append(f"sector(s) present on one side only but not recorded as orphans: {shown}")

    if failures:
        raise GateError(failures)
    return result
