"""End-to-end engine, the validation gate, and the command line."""

from __future__ import annotations

import dataclasses
import json
import math
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from attribution import io
from attribution.cli import main, safe_name, write_result
from attribution.engine import EngineConfig, run_attribution
from attribution.outputs import Result
from attribution.sample import generate_sample, write_sample
from attribution.schemas import CASH_KEY
from attribution.validate import GateError, validate_result

FAST = EngineConfig(portfolio_name="Sample PMS", bootstrap=False)


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    d = tmp_path_factory.mktemp("sample_data")
    write_sample(generate_sample(), d, "csv")
    return d


@pytest.fixture(scope="module")
def full(data_dir: Path) -> Result:
    return run_attribution(data_dir, config=EngineConfig(portfolio_name="Sample PMS"))


@pytest.fixture(scope="module")
def fast(data_dir: Path) -> Result:
    return run_attribution(data_dir, config=FAST)


def _copy_data(src: Path, dst: Path) -> Path:
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        (dst / f.name).write_bytes(f.read_bytes())
    return dst


# --- end to end --------------------------------------------------------------


def test_sample_produces_complete_valid_result(full: Result) -> None:
    assert validate_result(full) is full
    a = full.attribution
    assert a.period.n_periods == 60
    assert [l.name for l in a.decision_view.layers] == ["macro", "allocation", "selection", "interaction"]
    assert len(a.decision_view.by_sector) == 15 and len(a.factor_view.factors) == 6
    assert a.reconciliation.passes and full.reconciliation.passes
    rec = full.reconciliation
    for diff in (rec.decision_minus_active, rec.factor_minus_active, rec.decision_minus_factor):
        assert abs(diff) <= 1e-9
    assert rec.tolerance == 1e-9
    assert a.totals.active == pytest.approx(rec.active_return, abs=0)
    assert a.meta.synthetic_data is True
    assert set(full.run.inputs) >= {"holdings", "prices", "benchmark", "portfolio_returns", "policy"}
    assert all("sha256:" in v for k, v in full.run.inputs.items())


def test_both_views_equal_net_active(full: Result) -> None:
    a = full.attribution
    assert a.decision_view.total == pytest.approx(a.totals.active, abs=1e-9)
    assert a.factor_view.total == pytest.approx(a.totals.active, abs=1e-9)
    comps = a.decision_view.macro_anatomy.components
    assert set(comps) == {"exposure", "gross_to_net"}
    assert comps["exposure"] + comps["gross_to_net"] == pytest.approx(a.decision_view.layer("macro"), abs=1e-12)


def test_gross_net_bridge_is_first_class(full: Result) -> None:
    g = full.gross_to_net
    assert g.cost_cumulative == pytest.approx(g.net_cumulative - g.gross_cumulative, abs=1e-15)
    assert g.cost_annualised == pytest.approx(g.net_annualised - g.gross_annualised, abs=1e-15)
    assert g.consistent and g.label == "fees and trading costs"
    assert g.mean_abs_monthly_gap == pytest.approx(1.02 ** (1 / 12) - 1, abs=1e-10)  # sample: exactly the 2% fee
    assert full.attribution.totals.portfolio == pytest.approx(g.net_cumulative, abs=0)


def test_inconsistent_gross_net_is_flagged_not_called_fees(data_dir: Path, tmp_path: Path) -> None:
    d = _copy_data(data_dir, tmp_path / "bad")
    rep = io.load_portfolio_returns(d / "portfolio_returns.csv")
    rep["total_return"] = rep["total_return"] + 0.01  # 1% a month away from holdings x prices
    io.write_table(rep, d / "portfolio_returns.csv", io.PORTFOLIO_RETURN, ["SYNTHETIC SAMPLE DATA - tampered"])
    r = run_attribution(d, config=FAST)
    assert not r.gross_to_net.consistent
    assert "UNEXPLAINED" in r.gross_to_net.label
    lim = [x for x in r.limitations if x.id == "gross_net_inconsistent"]
    assert len(lim) == 1 and lim[0].severity == "material" and "NOT labelled as fees" in lim[0].text


def test_json_round_trips_unchanged(full: Result, tmp_path: Path) -> None:
    text = full.to_json()
    back = Result.from_json(text)
    assert back == full
    assert back.to_json() == text
    json_path, _ = write_result(full, tmp_path, "Sample PMS")
    assert Result.from_json(json_path.read_text(encoding="utf-8")) == full


def test_period_restriction(data_dir: Path) -> None:
    r = run_attribution(data_dir, period=("2022-01", "2025-12"), config=FAST)
    assert r.attribution.period.n_periods == 48
    assert str(r.attribution.period.start) == "2022-01-31" and str(r.attribution.period.end) == "2025-12-31"


# --- limitations -------------------------------------------------------------


def test_limitations_are_structured_and_mirrored(full: Result) -> None:
    assert all(x.severity in ("info", "caution", "material") and x.affects for x in full.limitations)
    ids = {x.id for x in full.limitations}
    assert len(ids) == len(full.limitations)
    assert len(full.attribution.limitations) == len(full.limitations)
    assert "synthetic_data" in ids and "market_vs_macro" in ids and "orphan_sectors" in ids
    assert "few_derisking_episodes" in ids  # the sample has a single crisis


def test_each_spanning_interval_is_named(full: Result) -> None:
    spans = {x.id for x in full.limitations if x.id.startswith("ci_spans_zero:")}
    assert spans, "the sample has intervals spanning zero; each must be listed"
    for x in full.limitations:
        if x.id.startswith("ci_spans_zero:"):
            assert x.id.split(":", 1)[1] in x.affects


def test_short_window_without_crisis_gets_its_limitations(data_dir: Path) -> None:
    r = run_attribution(data_dir, period=("2021-10", "2024-04"), config=FAST)  # 31 months, before the crisis
    ids = {x.id for x in r.limitations}
    assert {"short_history", "macro_never_fired", "few_derisking_episodes"} <= ids
    assert r.attribution.decision_view.macro_anatomy.components["exposure"] == 0.0


def test_collinear_factors_add_material_limitation(data_dir: Path, tmp_path: Path) -> None:
    d = _copy_data(data_dir, tmp_path / "coll")
    fr = io.load_factor_returns(d / "factor_returns.csv")
    wide = fr.pivot(index="date", columns="factor", values="total_return")
    wide["value"] = wide["size"] + 1e-4 * (pd.Series(range(len(wide)), index=wide.index) % 3 - 1)
    long = wide.reset_index().melt(id_vars="date", var_name="factor", value_name="total_return")
    io.write_table(long, d / "factor_returns.csv", io.FACTOR_RETURNS, ["SYNTHETIC - collinear test"])
    r = run_attribution(d, config=FAST)
    lim = [x for x in r.limitations if x.id == "factor_collinearity"]
    assert len(lim) == 1 and lim[0].severity == "material"


# --- the gate: one test per failure mode -------------------------------------


def _with(result: Result, **parts: object) -> Result:
    return dataclasses.replace(result, **parts)  # type: ignore[arg-type]


def test_gate_catches_broken_reconciliation(fast: Result) -> None:
    rec = dataclasses.replace(fast.reconciliation, decision_total=fast.reconciliation.decision_total + 0.01)
    with pytest.raises(GateError, match="reconciliation broken: decision view total vs active"):
        validate_result(_with(fast, reconciliation=rec))
    rec2 = dataclasses.replace(fast.reconciliation, factor_total=fast.reconciliation.factor_total - 1e-6)
    with pytest.raises(GateError, match="factor view total vs active"):
        validate_result(_with(fast, reconciliation=rec2))


def test_gate_catches_weights_not_summing_to_one(fast: Result) -> None:
    with pytest.raises(GateError, match="portfolio weights do not sum to 1.0"):
        validate_result(_with(fast, checks=dataclasses.replace(fast.checks, holdings_weight_max_deviation=1e-4)))
    with pytest.raises(GateError, match="benchmark weights do not sum to 1.0"):
        validate_result(_with(fast, checks=dataclasses.replace(fast.checks, benchmark_weight_max_deviation=2e-6)))


def test_gate_catches_non_finite_numbers(fast: Result) -> None:
    g = dataclasses.replace(fast.gross_to_net, max_abs_monthly_gap=math.nan)
    with pytest.raises(GateError, match=r"non-finite.*gross_to_net\.max_abs_monthly_gap"):
        validate_result(_with(fast, gross_to_net=g))
    rec = dataclasses.replace(fast.reconciliation, decision_components={**fast.reconciliation.decision_components,
                                                                         "allocation": math.inf})
    with pytest.raises(GateError, match="non-finite"):
        validate_result(_with(fast, reconciliation=rec))


def test_gate_catches_short_period(fast: Result, data_dir: Path) -> None:
    with pytest.raises(GateError, match="at least 24 are required"):
        validate_result(_with(fast, checks=dataclasses.replace(fast.checks, n_months=23)))
    with pytest.raises(GateError, match="12 months; at least 24"):
        run_attribution(data_dir, period=("2025-10", "2026-09"), config=FAST)


def test_gate_catches_policy_outside_its_band(fast: Result) -> None:
    chk = dataclasses.replace(fast.checks, policy_equity_weight=1.0, policy_equity_band=(0.6, 0.95))
    with pytest.raises(GateError, match="outside its own band"):
        validate_result(_with(fast, checks=chk))


def test_gate_catches_unrecorded_orphan_sector(fast: Result) -> None:
    assert fast.checks.benchmark_not_held, "sample should contain a benchmark-only sector (Energy)"
    chk = dataclasses.replace(fast.checks, orphans_recorded=())
    with pytest.raises(GateError, match="not recorded as orphans"):
        validate_result(_with(fast, checks=chk))
    chk2 = dataclasses.replace(fast.checks, held_not_in_benchmark=("2024-01 Realty",))
    with pytest.raises(GateError, match="2024-01 Realty"):
        validate_result(_with(fast, checks=chk2))


def test_gate_reports_every_failure_together(fast: Result) -> None:
    chk = dataclasses.replace(fast.checks, n_months=10, holdings_weight_max_deviation=1.0)
    with pytest.raises(GateError) as exc:
        validate_result(_with(fast, checks=chk))
    assert len(exc.value.failures) >= 2


def test_policy_with_cash_equity_band_passes(data_dir: Path) -> None:
    base = io.load_policy(data_dir / "policy.json")
    nw = dict(base.neutral_weights)
    biggest = max((k for k in nw if k != CASH_KEY), key=lambda k: nw[k])
    nw[biggest] = round(nw[biggest] - 0.10, 6)
    nw[CASH_KEY] = 0.10
    bands = {k: (max(0.0, v - 0.05), min(1.0, v + 0.05)) for k, v in nw.items()}
    bands[CASH_KEY] = (0.0, 0.40)
    pol = dataclasses.replace(base, neutral_weights=nw, bands=bands)
    r = run_attribution(data_dir, pol, config=FAST)
    assert r.checks.policy_equity_weight == pytest.approx(0.9)
    assert r.reconciliation.passes


# --- writing: never a partial or overwritten file ------------------------------


def test_broken_reconciliation_raises_and_writes_nothing(fast: Result, tmp_path: Path) -> None:
    out = tmp_path / "out"
    good, _ = write_result(fast, out, "Sample PMS")
    before = sorted(p.name for p in out.iterdir())
    rec = dataclasses.replace(fast.reconciliation, decision_total=fast.reconciliation.decision_total + 0.05)
    with pytest.raises(GateError):
        write_result(_with(fast, reconciliation=rec), out, "Sample PMS")
    assert sorted(p.name for p in out.iterdir()) == before  # nothing new, last good file intact
    assert Result.from_json(good.read_text(encoding="utf-8")) == fast


def test_two_runs_same_day_make_two_files(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "runs"
    args = ["run", "--data-dir", str(data_dir), "--out-dir", str(out), "--portfolio-name", "Sample PMS", "--no-bootstrap"]
    assert main(args) == 0
    assert main(args) == 0
    names = sorted(p.name for p in out.iterdir())
    assert names == [
        "attribution_Sample_PMS_2026-09-30.json", "attribution_Sample_PMS_2026-09-30.log",
        "attribution_Sample_PMS_2026-09-30_2.json", "attribution_Sample_PMS_2026-09-30_2.log",
    ]
    log = (out / "attribution_Sample_PMS_2026-09-30.log").read_text(encoding="utf-8")
    for needle in ("package version", "linking method", "factor order", "validation        PASSED", "sha256:", "UTC"):
        assert needle in log
    first = json.loads((out / "attribution_Sample_PMS_2026-09-30.json").read_text(encoding="utf-8"))
    assert first["attribution"]["reconciliation"]["passes"] is True


def test_cli_failure_writes_nothing(data_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "fail"
    code = main(["run", "--data-dir", str(data_dir), "--out-dir", str(out), "--portfolio-name", "X",
                 "--start", "2025-10", "--end", "2026-09", "--no-bootstrap"])
    assert code == 2
    assert not out.exists() or not any(out.iterdir())
    assert "VALIDATION GATE FAILED" in capsys.readouterr().err


def test_python_dash_m_entry_point(data_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "m"
    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "-m", "attribution", "run", "--data-dir", str(data_dir), "--out-dir", str(out),
         "--portfolio-name", "Sample PMS", "--no-bootstrap"],
        cwd=root, capture_output=True, text=True, env={"PYTHONPATH": str(root), "PATH": ""},
    )
    assert proc.returncode == 0, proc.stderr
    assert (out / "attribution_Sample_PMS_2026-09-30.json").exists()


def test_safe_name() -> None:
    assert safe_name("Acme PMS / Fund #1") == "Acme_PMS_Fund_1"
    with pytest.raises(ValueError):
        safe_name("///")
