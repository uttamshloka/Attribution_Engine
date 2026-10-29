from __future__ import annotations

import dataclasses
import json

import pytest
from conftest import make_result

from attribution.result import (
    AttributionResult,
    DecisionLayer,
    DecisionView,
    Reconciliation,
    ResultValidationError,
    SectorAttribution,
    Totals,
)


def test_json_round_trip(result: AttributionResult) -> None:
    text = result.to_json()
    again = AttributionResult.from_json(text)
    assert again == result
    assert again.to_json() == text


def test_json_top_level_structure(result: AttributionResult) -> None:
    data = json.loads(result.to_json())
    assert list(data) == [
        "meta", "period", "totals", "decision_view", "factor_view", "reconciliation", "limitations",
    ]
    assert set(data["totals"]) == {"portfolio", "benchmark", "active", "ci", "portfolio_gross", "active_gross"}
    assert set(data["decision_view"]) == {"layers", "by_sector", "macro_anatomy", "cost"}
    assert set(data["factor_view"]) == {"factors", "specific_return", "diagnostics"}
    assert set(data["reconciliation"]) == {"decision_total", "factor_total", "difference", "tolerance", "passes",
                                           "layers_total", "gross_active"}
    assert data["reconciliation"]["passes"] is True
    assert data["meta"]["schema_version"] == "1.1.0"


def test_views_reconcile_to_active(result: AttributionResult) -> None:
    assert result.decision_view.total == pytest.approx(result.totals.active, abs=1e-12)
    assert result.factor_view.total == pytest.approx(result.totals.active, abs=1e-12)


def test_inconsistent_passes_flag_rejected() -> None:
    with pytest.raises(ResultValidationError, match="passes"):
        make_result(passes_override=False)


def test_failing_reconciliation_is_representable(result: AttributionResult) -> None:
    rec = Reconciliation.compute(
        decision_total=result.reconciliation.decision_total,
        factor_total=result.reconciliation.decision_total - 0.001,
        active=result.totals.active,
    )
    assert rec.passes is False
    specific = dataclasses.replace(
        result.factor_view.specific_return, value=result.factor_view.specific_return.value - 0.001
    )
    fv = dataclasses.replace(result.factor_view, specific_return=specific)
    broken = dataclasses.replace(result, factor_view=fv, reconciliation=rec)
    assert AttributionResult.from_json(broken.to_json()).reconciliation.passes is False


def test_factor_total_must_match_contributions(result: AttributionResult) -> None:
    rec = Reconciliation.compute(result.reconciliation.decision_total, 0.5, result.totals.active)
    with pytest.raises(ResultValidationError, match="factor_total"):
        dataclasses.replace(result, reconciliation=rec)


def test_active_must_equal_portfolio_minus_benchmark(result: AttributionResult) -> None:
    with pytest.raises(ResultValidationError, match="portfolio - benchmark"):
        Totals(0.2, 0.1, 0.2, result.totals.ci)


def test_layers_must_be_the_four_in_order(result: AttributionResult) -> None:
    layers = result.decision_view.layers
    with pytest.raises(ResultValidationError, match="layers"):
        dataclasses.replace(result.decision_view, layers=(layers[1], layers[0], layers[2], layers[3]))
    with pytest.raises(ResultValidationError, match="unknown decision layer"):
        DecisionLayer("timing", 0.0)


def test_by_sector_must_sum_to_layers(result: AttributionResult) -> None:
    rows = list(result.decision_view.by_sector)
    rows[0] = dataclasses.replace(rows[0], selection=rows[0].selection + 0.01)
    with pytest.raises(ResultValidationError, match="selection"):
        DecisionView(result.decision_view.layers, tuple(rows), result.decision_view.macro_anatomy)


def test_macro_components_must_sum_to_macro(result: AttributionResult) -> None:
    anatomy = dataclasses.replace(result.decision_view.macro_anatomy, components={"fees": -0.01})
    with pytest.raises(ResultValidationError, match="macro_anatomy"):
        dataclasses.replace(result.decision_view, macro_anatomy=anatomy)


def test_unknown_sector_rejected() -> None:
    with pytest.raises(ResultValidationError, match="unknown sector"):
        SectorAttribution("Banks", 0, 0, 0, 0, 0, 0, 0)


def test_nan_rejected_on_construction_and_parse(result: AttributionResult) -> None:
    with pytest.raises(ResultValidationError, match="finite"):
        Totals(float("nan"), 0.1, 0.0, result.totals.ci)
    text = result.to_json().replace('"r_squared": 0.82', '"r_squared": NaN')
    with pytest.raises(ResultValidationError, match="non-finite"):
        AttributionResult.from_json(text)


def test_from_dict_rejects_unknown_and_missing_keys(result: AttributionResult) -> None:
    data = result.to_dict()
    data["totals"]["alpha"] = 0.0
    with pytest.raises(ResultValidationError, match="unexpected key"):
        AttributionResult.from_dict(data)
    data = result.to_dict()
    del data["period"]["n_periods"]
    with pytest.raises(ResultValidationError, match="missing key 'n_periods'"):
        AttributionResult.from_dict(data)


def test_from_dict_rejects_wrong_types(result: AttributionResult) -> None:
    data = result.to_dict()
    data["period"]["n_periods"] = 60.0
    with pytest.raises(ResultValidationError, match="integer"):
        AttributionResult.from_dict(data)
    data = result.to_dict()
    data["meta"]["frequency"] = "hourly"
    with pytest.raises(ResultValidationError, match="frequency"):
        AttributionResult.from_dict(data)
