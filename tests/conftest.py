from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest

from attribution.result import (
    AttributionResult,
    ConfidenceInterval,
    DecisionLayer,
    DecisionView,
    FactorContribution,
    FactorDiagnostics,
    FactorView,
    MacroAnatomy,
    Meta,
    Period,
    Reconciliation,
    SectorAttribution,
    SpecificReturn,
    Totals,
)
from attribution.sample import SampleData, generate_sample
from attribution.schemas import FACTORS


@pytest.fixture(scope="session")
def sample() -> SampleData:
    return generate_sample()


@pytest.fixture()
def holdings_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2025-01-31", "2025-01-31", "2025-02-28", "2025-02-28"],
            "security_id": ["A", "B", "A", "B"],
            "weight": [0.6, 0.4, 0.5, 0.5],
            "sector": ["Auto", "Bankex", "Auto", "Bankex"],
        }
    )


def make_result(passes_override: bool | None = None) -> AttributionResult:
    """A hand-built, internally consistent result (numbers are illustrative)."""
    by_sector = (
        SectorAttribution("Auto", 0.12, 0.08, 0.05, 0.04, 0.0012, 0.0008, 0.0004),
        SectorAttribution("Bankex", 0.18, 0.28, 0.02, 0.03, 0.0010, -0.0028, 0.0010),
        SectorAttribution("Capital Goods", 0.10, 0.05, 0.07, 0.05, 0.0006, 0.0010, 0.0010),
    )
    allocation = sum(s.allocation for s in by_sector)
    selection = sum(s.selection for s in by_sector)
    interaction = sum(s.interaction for s in by_sector)
    macro = -0.0040
    active = macro + allocation + selection + interaction

    factor_values = [0.0010, 0.0045, -0.0005, 0.0003, 0.0002, -0.0012]
    factors = tuple(
        FactorContribution(f, exposure=0.1 * (i + 1), factor_return=0.01, contribution=v, t_stat=1.5 if i else None)
        for i, (f, v) in enumerate(zip(FACTORS, factor_values, strict=True))
    )
    specific = active - sum(factor_values)
    benchmark = 0.1450
    rec = Reconciliation.compute(decision_total=active, factor_total=active, active=active)
    if passes_override is not None:
        rec = Reconciliation(rec.decision_total, rec.factor_total, rec.difference, rec.tolerance, passes_override)
    return AttributionResult(
        meta=Meta(
            portfolio_id="SAMPLE-PMS-01",
            portfolio_name="Sample PMS",
            benchmark_name="Synthetic BSE 40",
            currency="INR",
            frequency="monthly",
            linking="carino",
            engine_version="0.1.0",
            generated_at=datetime(2026, 10, 28, 19, 51, 0),
            synthetic_data=True,
        ),
        period=Period(date(2021, 10, 31), date(2026, 9, 30), 60),
        totals=Totals(
            portfolio=benchmark + active,
            benchmark=benchmark,
            active=active,
            ci=ConfidenceInterval(level=0.9, lower=active - 0.02, upper=active + 0.02, method="block bootstrap"),
        ),
        decision_view=DecisionView(
            layers=(
                DecisionLayer("macro", macro),
                DecisionLayer("allocation", allocation),
                DecisionLayer("selection", selection),
                DecisionLayer("interaction", interaction),
            ),
            by_sector=by_sector,
            macro_anatomy=MacroAnatomy(
                average_equity_exposure=0.97,
                policy_equity_exposure=1.0,
                cash_return=0.066,
                components={"exposure_timing": -0.0030, "cash_carry": -0.0010},
            ),
        ),
        factor_view=FactorView(
            factors=factors,
            specific_return=SpecificReturn(value=specific, t_stat=0.4, information_ratio=None),
            diagnostics=FactorDiagnostics(
                method="time-series OLS", n_obs=60, r_squared=0.82, tracking_error=0.041,
                residual_volatility=0.017, warnings=("illustrative",),
            ),
        ),
        reconciliation=rec,
        limitations=("Synthetic data.", "Arithmetic linking shown for illustration."),
    )


@pytest.fixture()
def result() -> AttributionResult:
    return make_result()
