"""The Streamlit page, deployment packaging and path settings."""

from __future__ import annotations

import ast
import math
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import streamlit_app as app  # noqa: E402
from attribution import settings  # noqa: E402
from attribution.outputs import Result  # noqa: E402

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest


@pytest.fixture(scope="module")
def loaded() -> app.LoadedResult:
    return app.load_result_file(app.find_committed_result(ROOT / "results"))


def _numbers(cell_html: str) -> float:
    text = re.sub(r"<[^>]+>", "", cell_html).replace("%", "").replace("pp", "").strip()
    return float(text)


# --- the committed result ------------------------------------------------------


def test_committed_result_exists_and_passes_the_gate(loaded: app.LoadedResult) -> None:
    path = app.find_committed_result(ROOT / "results")
    assert path.parent == ROOT / "results" and path.name.startswith("attribution_")
    assert isinstance(loaded.result, Result)
    assert len(loaded.sha256) == 64
    assert loaded.result.intervals and loaded.result.factor_detail is not None and loaded.result.rolling is not None


# --- the page renders ------------------------------------------------------------


def test_page_renders_without_raising() -> None:
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert [t.label for t in at.tabs] == ["Decision view", "Factor view"]
    text = " ".join(m.value for m in at.markdown)
    assert "SYNTHETIC DATA" in text  # the badge, always rendered when the flag is set
    assert "Reconciliation" in text and text.count(">ties<") == 3
    assert text.index("MATERIAL") < text.index("CAUTION") < text.index("INFO")
    assert any(b.label == "Recompute from source data" for b in at.button)
    assert "Result SHA-256" in text


def test_recompute_button_runs_the_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(settings.ENV_OUTPUT_DIR, str(tmp_path))
    at = AppTest.from_file(str(ROOT / "streamlit_app.py"), default_timeout=180).run()
    next(b for b in at.button if b.label == "Recompute from source data").click().run()
    assert not at.exception, [e.value for e in at.exception]
    text = " ".join(m.value for m in at.markdown)
    assert "live recomputation" in text
    assert list(tmp_path.glob("attribution_*.json"))


def test_page_functions_run_without_a_server(loaded: app.LoadedResult) -> None:
    r = loaded.result
    assert "SYNTHETIC DATA" in app.header_html(loaded)
    app.waterfall_chart(app.decision_rows(r)).to_dict()
    app.rolling_chart(r, ["market", "size"]).to_dict()
    assert len(app.diagnostics_frame(r)) >= 8
    for name in app.SECTOR_COLUMNS:
        assert "<table" in app.sector_table_html(r, name, True)


# --- formatting rules ------------------------------------------------------------


def test_one_formatting_rule() -> None:
    assert app.fmt(0.123456, "pct") == "+12.35%"
    assert app.fmt(-0.0123, "pct") == "-1.23%"
    assert app.fmt(0.37, "weight") == "37.00%"
    assert app.fmt(-0.0915, "active_weight") == "-9.15 pp"
    assert app.fmt(10.4123, "t") == "+10.41"
    assert app.fmt(0.68469, "exposure") == "+0.685"
    assert app.fmt(-1e-17, "active_weight") == "+0.00 pp"  # no negative zero
    assert app.fmt(float("nan"), "pct") == "n/a"
    assert 'class="neg"' in app.fmt_html(-0.01, "pct") and "neg" not in app.fmt_html(0.01, "pct")


def test_rounded_to_total_keeps_the_identity() -> None:
    import random

    rng = random.Random(1)
    for _ in range(200):
        parts = [rng.uniform(-0.3, 0.3) for _ in range(13)]
        shown = app.rounded_to_total(parts, sum(parts), 2)
        assert round(sum(shown) * 100, 2) == round(sum(parts) * 100, 2)
        assert all(abs(a - b) * 100 < 0.01 + 1e-9 for a, b in zip(shown, parts, strict=True))


def test_displayed_waterfall_adds_up(loaded: app.LoadedResult) -> None:
    rows = app.decision_rows(loaded.result)
    vals = [float(str(r["label"]).replace("%", "")) for r in rows]
    d = int(rows[0]["decimals"])  # type: ignore[arg-type]
    assert round(sum(vals[:4]), d) == vals[4]          # four layers = gross active
    assert round(vals[4] + vals[5], d) == vals[6]      # gross + fees = net active


def test_displayed_sector_table_adds_up(loaded: app.LoadedResult) -> None:
    html_ = app.sector_table_html(loaded.result)
    body = [r for r in re.findall(r"<tr>(.*?)</tr>", html_) if "<td" in r]
    total = re.findall(r"<tr class='subtotal'>(.*?)</tr>", html_)[0]
    grid = [[_numbers(c) for c in re.findall(r"<td class='num'>(.*?)</td>", row)[-4:]] for row in body]
    tot = [_numbers(c) for c in re.findall(r"<td class='num'>(.*?)</td>", total)[-4:]]
    for row in grid:
        assert round(sum(row[:3]), 2) == row[3]                 # each row: effects = total
    for j in range(4):
        assert round(sum(r[j] for r in grid), 2) == tot[j]      # each column sums to the total row
    layers = {l.name: l.value for l in loaded.result.attribution.decision_view.layers}
    assert tot[0] == round(layers["allocation"] * 100, 2)       # ties to the decision layers


def test_displayed_factor_table_adds_up(loaded: app.LoadedResult) -> None:
    html_ = app.factor_table_html(loaded.result)
    rows = re.findall(r"<tr[^>]*>(.*?)</tr>", html_)[1:]
    contrib = [_numbers(re.findall(r"<td class='num'>(.*?)</td>", r)[3]) for r in rows[:-1]]
    total = _numbers(re.findall(r"<td class='num'>(.*?)</td>", rows[-1])[0])
    decimals = len(str(contrib[0]).split(".")[1]) if "." in str(contrib[0]) else 0
    assert math.isclose(round(sum(contrib), max(decimals, 2)), total, abs_tol=1e-9)


def test_reconciliation_strip_shows_all_three_identities(loaded: app.LoadedResult) -> None:
    lines = app.reconciliation_lines(loaded.result)
    assert len(lines) == 3 and all(l["ties"] for l in lines)
    strip = app.reconciliation_html(loaded.result)
    for needle in ("gross active", "net active", "Factor contributions"):
        assert needle in strip


# --- packaging and paths -----------------------------------------------------------


def _source_files() -> list[Path]:
    return [*sorted((ROOT / "attribution").glob("*.py")), ROOT / "streamlit_app.py"]


def test_no_colab_or_drive_paths_anywhere() -> None:
    for path in _source_files():
        text = path.read_text(encoding="utf-8")
        for bad in ("/content", "google.colab", "MyDrive"):
            assert bad not in text, f"{path.name} contains {bad!r}"


def test_requirements_are_pinned_runtime_imports_only() -> None:
    reqs = {}
    for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        assert re.fullmatch(r"[A-Za-z0-9_.-]+==[0-9][A-Za-z0-9.]*", line), f"not pinned exactly: {line}"
        name, version = line.split("==")
        reqs[name.lower()] = version
    imported: set[str] = set()
    for path in _source_files():
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                imported.add(node.module.split(".")[0])
    third_party = {m.lower() for m in imported if m not in sys.stdlib_module_names and m != "attribution"}
    assert third_party == set(reqs), f"imports {sorted(third_party)} vs requirements {sorted(reqs)}"
    assert "pytest" not in reqs


def test_settings_default_and_env_override(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for var in (settings.ENV_DATA_DIR, settings.ENV_RESULTS_DIR, settings.ENV_OUTPUT_DIR):
        monkeypatch.delenv(var, raising=False)
    assert settings.data_dir() == ROOT / "sample_data"
    assert settings.results_dir() == ROOT / "results"
    assert ROOT not in settings.output_dir().parents
    monkeypatch.setenv(settings.ENV_DATA_DIR, str(tmp_path / "d"))
    monkeypatch.setenv(settings.ENV_OUTPUT_DIR, str(tmp_path / "o"))
    assert settings.data_dir() == tmp_path / "d" and settings.output_dir() == tmp_path / "o"


def test_repository_layout_is_deployable() -> None:
    for rel in ("streamlit_app.py", "requirements.txt", ".streamlit/config.toml", "README.md",
                "sample_data/holdings.csv", "sample_data/policy.json"):
        assert (ROOT / rel).exists(), rel
    config = (ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8")
    assert 'base = "dark"' in config and "primaryColor" in config
