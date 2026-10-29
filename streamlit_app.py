"""Return attribution page (Streamlit).

Run locally with ``streamlit run streamlit_app.py``; on Streamlit Community
Cloud this file is the entry point.

Why the page loads a PRE-COMPUTED result by default
---------------------------------------------------
Streamlit Cloud starts and stops containers freely: a visitor often lands on a
cold container. A full engine run (decision view, factor model and 2,000-sample
bootstraps per line) takes several seconds there, so recomputing on every page
load would make the demo look broken. The page therefore reads the committed
result file in ``results/`` and offers a clearly labelled "Recompute from source
data" button for a live run.

Rules this page follows
-----------------------
* Every figure comes from the :class:`attribution.outputs.Result` JSON.
* Every number is formatted by :func:`fmt` (and :func:`fmt_html` for colour).
* When rounding would stop displayed components from visibly summing to their
  total, more decimals are shown (:func:`identity_decimals`).
* All page logic lives in functions, so tests can call them without a server.
"""

from __future__ import annotations

import hashlib
import html
import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import altair as alt
import pandas as pd
import streamlit as st

from attribution.outputs import Result
from attribution.settings import data_dir, output_dir, results_dir

__all__ = [
    "LoadedResult",
    "fmt",
    "fmt_html",
    "identity_decimals",
    "rounded_to_total",
    "find_committed_result",
    "load_result_file",
    "recompute",
    "decision_rows",
    "waterfall_chart",
    "layer_interval_html",
    "sector_frame",
    "sector_table_html",
    "factor_table_html",
    "rolling_frame",
    "rolling_chart",
    "diagnostics_frame",
    "reconciliation_lines",
    "reconciliation_html",
    "limitations_html",
    "header_html",
    "main",
]

# ---------------------------------------------------------------------------
# Palette (dark, restrained; one accent; muted red for negatives)
# ---------------------------------------------------------------------------
ACCENT = "#7A9CC6"
NEGATIVE = "#C27C7C"
MUTED = "#8A9099"
SUBTOTAL = "#C9CED6"
AMBER = "#C9A227"
GRID = "#2A2F36"

Kind = Literal["pct", "weight", "active_weight", "t", "exposure", "diff", "ratio", "count"]

# ---------------------------------------------------------------------------
# The ONE formatting helper
# ---------------------------------------------------------------------------


def fmt(value: float | int | None, kind: Kind, decimals: int | None = None) -> str:
    """Format any figure on the page.

    pct            percentage, explicit sign, 2 decimals:      +12.34%
    weight         percentage weight, 2 decimals:               37.00%
    active_weight  signed weight difference in points, 2 dp:    +1.49 pp
    t              t-statistic, explicit sign, 2 decimals:      +10.41
    exposure       factor exposure, explicit sign, 3 decimals:  +0.685
    diff           reconciliation difference, scientific:      +1.33e-15
    ratio          unsigned statistic, 3 decimals:              0.715
    count          integer:                                     60
    """
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "n/a"
    v = float(value)
    if kind in ("pct", "weight", "active_weight", "t", "exposure", "ratio"):
        default = {"pct": 2, "weight": 2, "active_weight": 2, "t": 2, "exposure": 3, "ratio": 3}[kind]
        scale = 100.0 if kind in ("pct", "weight", "active_weight") else 1.0
        if round(v * scale, default if decimals is None else decimals) == 0:
            v = 0.0  # never show "-0.00"
    if kind == "pct":
        d = 2 if decimals is None else decimals
        return f"{v * 100:+.{d}f}%"
    if kind == "weight":
        d = 2 if decimals is None else decimals
        return f"{v * 100:.{d}f}%"
    if kind == "active_weight":
        d = 2 if decimals is None else decimals
        return f"{v * 100:+.{d}f} pp"
    if kind == "t":
        return f"{v:+.{2 if decimals is None else decimals}f}"
    if kind == "exposure":
        return f"{v:+.{3 if decimals is None else decimals}f}"
    if kind == "diff":
        return f"{v:+.2e}"
    if kind == "ratio":
        return f"{v:.{3 if decimals is None else decimals}f}"
    if kind == "count":
        return f"{int(round(v))}"
    raise ValueError(f"unknown format kind {kind!r}")


def fmt_html(value: float | int | None, kind: Kind, decimals: int | None = None) -> str:
    """:func:`fmt`, wrapped so negative numbers show in muted red."""
    text = html.escape(fmt(value, kind, decimals))
    if value is not None and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) < 0:
        if kind not in ("diff",):
            return f'<span class="neg">{text}</span>'
    return text


def identity_decimals(parts: Sequence[float], total: float, base: int = 2, max_decimals: int = 6) -> int:
    """Fewest decimals (in percent) at which the rounded parts visibly sum to the rounded total."""
    for d in range(base, max_decimals + 1):
        shown = sum(round(p * 100, d) for p in parts)
        if round(shown, d) == round(total * 100, d):
            return d
    return max_decimals


def rounded_to_total(parts: Sequence[float], total: float, decimals: int = 2) -> list[float]:
    """Round ``parts`` (in percent, to ``decimals``) so they add up exactly to the rounded ``total``.

    Largest-remainder rounding: each part is first rounded down, then the units
    still missing are given to the parts with the largest remainders. Every
    shown value stays within one unit of its exact value, and the displayed
    components always sum to the displayed total. Returns decimal fractions.
    """
    unit = 10.0 ** (-decimals)
    scaled = [p * 100 / unit for p in parts]
    target = round(total * 100 / unit)
    floors = [math.floor(x) for x in scaled]
    short = int(target - sum(floors))
    order = sorted(range(len(scaled)), key=lambda i: scaled[i] - floors[i], reverse=(short > 0))
    out = list(floors)
    for i in order[: abs(short)]:
        out[i] += 1 if short > 0 else -1
    return [v * unit / 100 for v in out]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoadedResult:
    result: Result
    sha256: str
    source: str          # human label for where the result came from
    path: str | None     # file it was read from or written to


def find_committed_result(folder: Path | None = None) -> Path:
    folder = folder or results_dir()
    files = sorted(folder.glob("attribution_*.json"))
    if not files:
        raise FileNotFoundError(f"no committed result file (attribution_*.json) found in {folder}")
    return files[-1]


def load_result_file(path: Path) -> LoadedResult:
    """Read a result JSON, check it against the validation gate, and hash the file."""
    from attribution.validate import validate_result

    raw = Path(path).read_bytes()
    result = validate_result(Result.from_json(raw.decode("utf-8")))
    return LoadedResult(result, hashlib.sha256(raw).hexdigest(), f"committed result file {Path(path).name}", str(path))


def recompute(portfolio_name: str) -> LoadedResult:
    """Live run of the engine on the committed source data; the result is written to OUTPUT_DIR."""
    from attribution.cli import write_result
    from attribution.engine import EngineConfig, run_attribution

    result = run_attribution(data_dir(), config=EngineConfig(portfolio_name=portfolio_name))
    json_path, _ = write_result(result, output_dir(), portfolio_name)
    digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return LoadedResult(result, digest, f"live recomputation at {when}", str(json_path))


@st.cache_resource(show_spinner=False)
def _cached_committed(path_str: str, mtime: float) -> LoadedResult:
    return load_result_file(Path(path_str))


# ---------------------------------------------------------------------------
# Decision view
# ---------------------------------------------------------------------------

_LAYER_LABELS = {
    "macro": "Macro exposure timing",
    "allocation": "Allocation",
    "selection": "Selection",
    "interaction": "Interaction",
}


def decision_rows(result: Result) -> list[dict[str, object]]:
    """Waterfall rows: four components, = gross active, fees and trading, = net active."""
    dv = result.attribution.decision_view
    rows: list[dict[str, object]] = []
    running = 0.0
    for layer in dv.layers:
        rows.append({"step": _LAYER_LABELS[layer.name], "kind": "component", "value": layer.value,
                     "start": running, "end": running + layer.value, "line": f"decision_view.{layer.name}"})
        running += layer.value
    rows.append({"step": "= Gross active", "kind": "subtotal", "value": dv.total, "start": 0.0, "end": dv.total,
                 "line": "totals.active_gross"})
    rows.append({"step": "Fees and trading costs", "kind": "component", "value": dv.cost, "start": dv.total,
                 "end": dv.total + dv.cost, "line": "decision_view.cost"})
    rows.append({"step": "= Net active", "kind": "subtotal", "value": dv.net_total, "start": 0.0,
                 "end": dv.net_total, "line": "totals.active"})
    d1 = identity_decimals([layer.value for layer in dv.layers], dv.total)
    d2 = identity_decimals([dv.total, dv.cost], dv.net_total)
    d = max(d1, d2)
    for r in rows:
        r["label"] = fmt(float(r["value"]), "pct", d)  # type: ignore[arg-type]
        r["decimals"] = d
    return rows


def waterfall_chart(rows: list[dict[str, object]]) -> alt.LayerChart:
    df = pd.DataFrame(rows)
    df["lo"] = df[["start", "end"]].min(axis=1) * 100
    df["hi"] = df[["start", "end"]].max(axis=1) * 100
    df["shade"] = [
        "subtotal" if k == "subtotal" else ("negative" if v < 0 else "positive")
        for k, v in zip(df["kind"], df["value"], strict=True)
    ]
    order = list(df["step"])
    base = alt.Chart(df).encode(
        x=alt.X("step:N", sort=order, title="Decision layer",
                axis=alt.Axis(labelAngle=0, labelOverlap=False, labelExpr="split(datum.label, ' ')",
                              labelColor=SUBTOTAL, titleColor=MUTED, labelPadding=6)),
    )
    bars = base.mark_bar(size=46).encode(
        y=alt.Y("lo:Q", title="Contribution to cumulative active return (%)",
                axis=alt.Axis(labelColor=SUBTOTAL, titleColor=MUTED, gridColor=GRID)),
        y2="hi:Q",
        color=alt.Color("shade:N", legend=None,
                        scale=alt.Scale(domain=["positive", "negative", "subtotal"], range=[ACCENT, NEGATIVE, SUBTOTAL])),
        tooltip=[alt.Tooltip("step:N", title="Line"), alt.Tooltip("label:N", title="Contribution")],
    )
    labels = base.mark_text(dy=-8, color=SUBTOTAL, fontSize=12).encode(y=alt.Y("hi:Q"), text="label:N")
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(y="y:Q")
    return (bars + labels + zero).properties(height=340).configure_view(stroke=None).configure(background="transparent")


def layer_interval_html(result: Result) -> str:
    """Each decision line with its contribution and its bootstrap interval; zero-spanning intervals marked."""
    rows = decision_rows(result)
    d = int(rows[0]["decimals"])  # type: ignore[arg-type]
    out = ['<table class="att"><thead><tr><th>Line</th><th class="num">Contribution</th>'
           '<th class="num">95% interval (annualised)</th><th></th></tr></thead><tbody>']
    for r in rows:
        ci = result.interval(str(r["line"]))
        if ci is not None:
            interval = f"{fmt_html(ci.lower, 'pct')} to {fmt_html(ci.upper, 'pct')}"
            marker = '<span class="marker">spans zero</span>' if ci.spans_zero else ""
        else:
            interval, marker = '<span class="muted">not estimated</span>', ""
        cls = "subtotal" if r["kind"] == "subtotal" else ""
        out.append(f'<tr class="{cls}"><td>{html.escape(str(r["step"]))}</td>'
                   f'<td class="num">{fmt_html(float(r["value"]), "pct", d)}</td>'  # type: ignore[arg-type]
                   f'<td class="num">{interval}</td><td>{marker}</td></tr>')
    out.append("</tbody></table>")
    out.append('<p class="note">Contributions are cumulative and linked; intervals are on the annualised mean of '
               'monthly contributions, so the two are on different bases.</p>')
    return "".join(out)


SECTOR_COLUMNS = {
    "Sector": "sector",
    "Portfolio weight": "w_p",
    "Benchmark weight": "w_b",
    "Active weight": "active",
    "Allocation": "allocation",
    "Selection": "selection",
    "Interaction": "interaction",
    "Total": "total",
}


def sector_frame(result: Result) -> pd.DataFrame:
    rows = []
    for s in result.attribution.decision_view.by_sector:
        rows.append({
            "sector": s.sector, "w_p": s.portfolio_weight, "w_b": s.benchmark_weight,
            "active": s.portfolio_weight - s.benchmark_weight, "allocation": s.allocation,
            "selection": s.selection, "interaction": s.interaction,
            "total": s.allocation + s.selection + s.interaction,
        })
    return pd.DataFrame(rows)


def _bar(value: float, max_abs: float) -> str:
    width = 0.0 if max_abs <= 0 else min(1.0, abs(value) / max_abs) * 50.0
    side = "left:50%" if value >= 0 else f"left:{50.0 - width:.2f}%"
    colour = ACCENT if value >= 0 else NEGATIVE
    return (f'<div class="bar"><div class="axis"></div>'
            f'<div class="fill" style="{side};width:{width:.2f}%;background:{colour}"></div></div>')


def sector_table_html(result: Result, sort_by: str = "Total", descending: bool = True) -> str:
    df = sector_frame(result)
    key = SECTOR_COLUMNS[sort_by]
    df = df.sort_values(key, ascending=not descending, kind="mergesort").reset_index(drop=True)
    effects = ["allocation", "selection", "interaction"]
    col_totals = [float(df[c].sum()) for c in effects]
    # Display rounding that keeps every identity in the table: each effect column is rounded so it sums
    # exactly to its rounded total; each row's Total is the sum of its displayed effects.
    d = 2
    shown = df.copy()
    for c, tot in zip(effects, col_totals, strict=True):
        shown[c] = rounded_to_total(df[c].tolist(), tot, d)
    shown["total"] = shown[effects].sum(axis=1)
    col_shown = [round(float(shown[c].sum()), 10) for c in effects]
    df = shown
    max_abs = float(df["active"].abs().max()) or 1.0
    head = "".join(f'<th class="{"num" if c != "Sector" else ""}">{c}</th>' for c in SECTOR_COLUMNS)
    out = [f'<table class="att sector"><thead><tr>{head}</tr></thead><tbody>']
    for r in df.itertuples(index=False):
        out.append(
            f"<tr><td>{html.escape(r.sector)}</td><td class='num'>{fmt_html(r.w_p, 'weight')}</td>"
            f"<td class='num'>{fmt_html(r.w_b, 'weight')}</td>"
            f"<td class='num'><div class='activecell'>{_bar(r.active, max_abs)}"
            f"<span>{fmt_html(r.active, 'active_weight')}</span></div></td>"
            f"<td class='num'>{fmt_html(r.allocation, 'pct', d)}</td><td class='num'>{fmt_html(r.selection, 'pct', d)}</td>"
            f"<td class='num'>{fmt_html(r.interaction, 'pct', d)}</td><td class='num'>{fmt_html(r.total, 'pct', d)}</td></tr>"
        )
    w_p, w_b = float(df["w_p"].sum()), float(df["w_b"].sum())
    out.append(
        f"<tr class='subtotal'><td>Total</td><td class='num'>{fmt_html(w_p, 'weight')}</td>"
        f"<td class='num'>{fmt_html(w_b, 'weight')}</td><td class='num'>{fmt_html(w_p - w_b, 'active_weight')}</td>"
        + "".join(f"<td class='num'>{fmt_html(v, 'pct', d)}</td>" for v in col_shown)
        + f"<td class='num'>{fmt_html(sum(col_shown), 'pct', d)}</td></tr>"
    )
    out.append("</tbody></table>")
    out.append('<p class="note">Weights are averages over the period. Effects are linked, cumulative and scaled by the '
               'equity weight; they sum to the allocation, selection and interaction layers above. Effects are rounded '
               'so that every row and column adds up exactly as shown (largest-remainder rounding); each figure is '
               'within 0.01 percentage points of its exact value.</p>')
    return "".join(out)


# ---------------------------------------------------------------------------
# Factor view
# ---------------------------------------------------------------------------


def factor_table_html(result: Result) -> str:
    fv = result.attribution.factor_view
    detail = result.factor_detail
    parts = [f.contribution for f in fv.factors] + [fv.specific_return.value]
    d = identity_decimals(parts, fv.total)
    out = ['<table class="att"><thead><tr><th>Factor</th><th class="num">Exposure</th>'
           '<th class="num">Exposure 95% interval</th><th class="num">t-statistic</th>'
           '<th class="num">Contribution</th><th class="num">Contribution 95% interval (annualised)</th>'
           '<th></th></tr></thead><tbody>']

    def interval_cells(line: str) -> tuple[str, str]:
        ci = result.interval(line)
        if ci is None:
            return '<span class="muted">not estimated</span>', ""
        marker = '<span class="marker">spans zero</span>' if ci.spans_zero else ""
        return f"{fmt_html(ci.lower, 'pct')} to {fmt_html(ci.upper, 'pct')}", marker

    for f in fv.factors:
        if detail is not None and f.factor in detail.exposure_lower:
            e_int = f"{fmt_html(detail.exposure_lower[f.factor], 'exposure')} to {fmt_html(detail.exposure_upper[f.factor], 'exposure')}"
        else:
            e_int = '<span class="muted">not estimated</span>'
        c_int, marker = interval_cells(f"factor_view.{f.factor}")
        out.append(
            f"<tr><td>{html.escape(f.factor.replace('_', ' '))}</td><td class='num'>{fmt_html(f.exposure, 'exposure')}</td>"
            f"<td class='num'>{e_int}</td><td class='num'>{fmt_html(f.t_stat, 't')}</td>"
            f"<td class='num'>{fmt_html(f.contribution, 'pct', d)}</td><td class='num'>{c_int}</td><td>{marker}</td></tr>"
        )
    s_int, s_marker = interval_cells("factor_view.specific_return")
    sr = fv.specific_return
    out.append(
        f"<tr class='separated'><td>Specific return</td><td class='num'></td><td class='num'></td>"
        f"<td class='num'>{fmt_html(sr.t_stat, 't')}</td><td class='num'>{fmt_html(sr.value, 'pct', d)}</td>"
        f"<td class='num'>{s_int}</td><td>{s_marker}</td></tr>"
    )
    out.append(f"<tr class='subtotal'><td>= Net active</td><td></td><td></td><td></td>"
               f"<td class='num'>{fmt_html(fv.total, 'pct', d)}</td><td></td><td></td></tr>")
    out.append("</tbody></table>")
    order = " → ".join(f.replace("_", " ") for f in detail.order) if detail is not None else ""
    out.append('<p class="note">Exposures are coefficients from one full-period regression of monthly net active '
               f'return on factors orthogonalised in the order {html.escape(order)}; each contribution is incremental '
               'to the factors before it. Contributions are linked and cumulative.</p>')
    return "".join(out)


def rolling_frame(result: Result, factors: Sequence[str]) -> pd.DataFrame:
    roll = result.rolling
    if roll is None:
        return pd.DataFrame(columns=["window_end", "factor", "exposure"])
    rows = [
        {"window_end": pd.Timestamp(d), "factor": f.replace("_", " "), "exposure": roll.exposures[f][i]}
        for f in factors if f in roll.exposures
        for i, d in enumerate(roll.window_end)
    ]
    return pd.DataFrame(rows)


def rolling_chart(result: Result, factors: Sequence[str]) -> alt.Chart:
    df = rolling_frame(result, factors)
    window = result.rolling.window_months if result.rolling is not None else 0
    chart = alt.Chart(df).mark_line(strokeWidth=2).encode(
        x=alt.X("window_end:T", title=f"Window end (month), {window}-month rolling window",
                axis=alt.Axis(format="%b %Y", labelColor=SUBTOTAL, titleColor=MUTED, gridColor=GRID)),
        y=alt.Y("exposure:Q", title="Exposure (regression coefficient, unitless)",
                axis=alt.Axis(labelColor=SUBTOTAL, titleColor=MUTED, gridColor=GRID)),
        color=alt.Color("factor:N", title="Factor",
                        scale=alt.Scale(range=[ACCENT, SUBTOTAL, AMBER, NEGATIVE, "#8FB996", "#A68BC1"])),
        tooltip=[alt.Tooltip("factor:N", title="Factor"), alt.Tooltip("window_end:T", title="Window end", format="%b %Y"),
                 alt.Tooltip("exposure:Q", title="Exposure", format="+.3f")],
    )
    zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color=MUTED, strokeDash=[3, 3]).encode(y="y:Q")
    return (chart + zero).properties(height=320).configure_view(stroke=None).configure(background="transparent")


def diagnostics_frame(result: Result) -> pd.DataFrame:
    d = result.factor_detail
    fv = result.attribution.factor_view.diagnostics
    if d is None:
        return pd.DataFrame({"Measure": ["Method"], "Value": [fv.method]})
    rows = [
        ("Observations (months)", fmt(d.n_obs, "count")),
        ("R-squared", fmt(d.r_squared, "ratio")),
        ("Adjusted R-squared", fmt(d.adj_r_squared, "ratio")),
        ("Tracking error (annualised)", fmt(d.tracking_error, "weight")),
        ("Residual volatility (annualised)", fmt(d.residual_volatility, "weight")),
        ("Durbin-Watson", fmt(d.durbin_watson, "ratio")),
        ("Condition number of X'X, raw factors", fmt(d.condition_number_raw, "ratio")),
        ("Condition number of X'X, as used", fmt(d.condition_number_used, "ratio")),
        *[(f"VIF raw / as used: {k.replace('_', ' ')}", f"{fmt(v, 'ratio')} / {fmt(d.vif_used[k], 'ratio')}")
          for k, v in d.vif_raw.items()],
        ("Orthogonalisation order", " → ".join(d.order)),
        ("Method", fv.method),
    ]
    return pd.DataFrame(rows, columns=["Measure", "Value"])


# ---------------------------------------------------------------------------
# Reconciliation, limitations, header
# ---------------------------------------------------------------------------


def reconciliation_lines(result: Result) -> list[dict[str, object]]:
    rec = result.reconciliation
    tol = rec.tolerance
    lines = [
        ("Four decision layers = gross active", rec.decision_layers_total, rec.gross_active),
        ("Gross active + fees and trading = net active", rec.gross_active + rec.cost_line, rec.active_return),
        ("Factor contributions + specific return = net active", rec.factor_total, rec.active_return),
    ]
    return [{"identity": name, "left": a, "right": b, "difference": a - b, "ties": abs(a - b) <= tol, "tolerance": tol}
            for name, a, b in lines]


def reconciliation_html(result: Result) -> str:
    out = ['<div class="recon"><div class="recon-title">Reconciliation</div><table class="att"><thead><tr>'
           '<th>Identity</th><th class="num">Left side</th><th class="num">Right side</th>'
           '<th class="num">Difference</th><th>Status</th></tr></thead><tbody>']
    for r in reconciliation_lines(result):
        status = "ties" if r["ties"] else '<span class="neg">does not tie</span>'
        out.append(f"<tr><td>{html.escape(str(r['identity']))}</td>"
                   f"<td class='num'>{fmt_html(float(r['left']), 'pct', 6)}</td>"  # type: ignore[arg-type]
                   f"<td class='num'>{fmt_html(float(r['right']), 'pct', 6)}</td>"  # type: ignore[arg-type]
                   f"<td class='num'>{fmt(float(r['difference']), 'diff')}</td><td>{status}</td></tr>")  # type: ignore[arg-type]
    tol = result.reconciliation.tolerance
    out.append(f"</tbody></table><p class='note'>Tolerance {fmt(tol, 'diff')} on cumulative linked returns. The decision "
               f"view is built on gross returns and bridged to net by the fees line; the factor view is built on net "
               f"returns. Linking: {html.escape(result.reconciliation.linking)}.</p></div>")
    return "".join(out)


def limitations_html(result: Result) -> str:
    out = ['<div class="lims"><div class="recon-title">Limitations</div>']
    for severity in ("material", "caution", "info"):
        items = result.limitations_by_severity(severity)
        if not items:
            continue
        out.append(f'<div class="sev sev-{severity}">{severity.upper()} ({len(items)})</div><ul>')
        for x in items:
            out.append(f"<li><span>{html.escape(x.text)}</span>"
                       f"<span class='affects'>Affects: {html.escape(', '.join(x.affects))}</span></li>")
        out.append("</ul>")
    out.append("</div>")
    return "".join(out)


def header_html(loaded: LoadedResult) -> str:
    a = loaded.result.attribution
    m = a.meta
    badge = '<span class="badge">SYNTHETIC DATA</span>' if m.synthetic_data else ""
    return (
        f'<div class="band"><div class="band-top"><span class="pname">{html.escape(m.portfolio_name)}</span>{badge}</div>'
        f'<div class="band-mid">Benchmark {html.escape(m.benchmark_name)} &nbsp;·&nbsp; Period {a.period.start} to '
        f'{a.period.end} ({fmt(a.period.n_periods, "count")} months) &nbsp;·&nbsp; Currency {html.escape(m.currency)}'
        f' &nbsp;·&nbsp; Linking {html.escape(m.linking.capitalize())}</div>'
        f'<div class="band-small">As of {a.period.end} &nbsp;·&nbsp; Result SHA-256 {loaded.sha256} &nbsp;·&nbsp; '
        f'Source: {html.escape(loaded.source)} &nbsp;·&nbsp; Engine {html.escape(m.engine_version)}, '
        f'result schema {html.escape(m.schema_version)}</div></div>'
    )


def headline_html(result: Result) -> str:
    t = result.attribution.totals
    g = result.gross_to_net
    cells = [
        ("Portfolio, net of fees", fmt_html(t.portfolio, "pct")),
        ("Policy benchmark", fmt_html(t.benchmark, "pct")),
        ("Net active return", fmt_html(t.active, "pct")),
        ("Gross active return", fmt_html(t.active_gross, "pct")),
        ("Fees and trading costs", fmt_html(g.cost_cumulative, "pct")),
    ]
    inner = "".join(f'<div class="kpi"><div class="kpi-label">{k}</div><div class="kpi-value">{v}</div></div>'
                    for k, v in cells)
    return f'<div class="kpis">{inner}</div><p class="note">Cumulative over the period.</p>'


_CSS = f"""
<style>
.block-container {{ padding-top: 3.2rem; max-width: 1280px; }}
.band {{ border-bottom: 1px solid {GRID}; padding-bottom: .8rem; margin-bottom: 1rem; }}
.band-top {{ display:flex; align-items:center; gap:.9rem; }}
.pname {{ font-size: 1.55rem; font-weight: 600; letter-spacing: .01em; }}
.badge {{ border: 1px solid {AMBER}; color: {AMBER}; font-size: .72rem; font-weight: 600; letter-spacing: .12em;
          padding: .18rem .55rem; border-radius: 3px; }}
.band-mid {{ color: {SUBTOTAL}; font-size: .92rem; margin-top: .3rem; }}
.band-small {{ color: {MUTED}; font-size: .72rem; margin-top: .35rem; word-break: break-all; }}
.kpis {{ display:flex; gap: 2.2rem; flex-wrap: wrap; margin: .2rem 0 .1rem 0; }}
.kpi-label {{ color:{MUTED}; font-size:.75rem; text-transform: uppercase; letter-spacing:.06em; }}
.kpi-value {{ font-size:1.25rem; font-variant-numeric: tabular-nums; }}
table.att {{ width:100%; border-collapse: collapse; font-size: .86rem; font-variant-numeric: tabular-nums; }}
table.att, table.att th, table.att td {{ border-left:none !important; border-right:none !important; border-top:none; }}
table.att th {{ color:{MUTED}; font-weight:500; text-align:left; border-bottom:1px solid {GRID}; padding:.38rem .5rem;
               background:transparent; }}
table.att td {{ border-bottom:1px solid #1E2228; padding:.34rem .5rem; }}
table.att .num {{ text-align:right; white-space:nowrap; }}
table.att tr.subtotal td {{ border-top:1px solid {MUTED}; font-weight:600; color:{SUBTOTAL}; }}
table.att tr.separated td {{ border-top: 2px solid {GRID}; }}
.neg {{ color:{NEGATIVE}; }}
.muted {{ color:{MUTED}; }}
.marker {{ color:{MUTED}; font-size:.72rem; border:1px solid {GRID}; border-radius:3px; padding:.05rem .35rem;
           white-space:nowrap; }}
.note {{ color:{MUTED}; font-size:.75rem; margin-top:.35rem; }}
.activecell {{ display:flex; align-items:center; justify-content:flex-end; gap:.5rem; }}
.bar {{ position:relative; width:84px; height:8px; background:#1A1E24; }}
.bar .axis {{ position:absolute; left:50%; top:-2px; width:1px; height:12px; background:{MUTED}; }}
.bar .fill {{ position:absolute; top:0; height:8px; }}
.recon, .lims {{ border-top:1px solid {GRID}; margin-top:1.4rem; padding-top:.8rem; }}
.recon-title {{ font-size:.8rem; text-transform:uppercase; letter-spacing:.08em; color:{MUTED}; margin-bottom:.4rem; }}
.sev {{ font-size:.78rem; letter-spacing:.06em; margin-top:.6rem; }}
.sev-material {{ color:{AMBER}; }} .sev-caution {{ color:{SUBTOTAL}; }} .sev-info {{ color:{MUTED}; }}
.lims ul {{ margin:.2rem 0 .4rem 1rem; padding:0; }}
.lims li {{ margin:.25rem 0; font-size:.85rem; }}
.affects {{ display:block; color:{MUTED}; font-size:.72rem; }}
</style>
"""


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _current() -> LoadedResult:
    """The committed result, unless the user asked for a live recomputation in this session."""
    live = st.session_state.get("live_result")
    # Streamlit re-runs this script on every interaction, which re-creates the LoadedResult class; an object
    # stored in session state belongs to the previous run's class, so check its shape, not isinstance.
    if live is not None and all(hasattr(live, k) for k in ("result", "sha256", "source", "path")):
        return LoadedResult(live.result, live.sha256, live.source, live.path)
    path = find_committed_result()
    return _cached_committed(str(path), path.stat().st_mtime)


def render_controls(loaded: LoadedResult) -> None:
    left, right = st.columns([5, 2])
    with left:
        st.markdown(f'<p class="note">Showing: {html.escape(loaded.source)}.</p>', unsafe_allow_html=True)
    with right:
        if st.button("Recompute from source data", help="Runs the full engine on the committed input files. "
                     "Takes several seconds; the result is written to a temporary folder."):
            from attribution.validate import GateError

            try:
                with st.spinner("Running the engine on the source data..."):
                    st.session_state["live_result"] = recompute(loaded.result.attribution.meta.portfolio_name)
                st.rerun()
            except GateError as exc:
                st.error(f"The validation gate refused the recomputed result; the committed result is still shown.\n\n{exc}")
        if "live_result" in st.session_state and st.button("Show committed result"):
            del st.session_state["live_result"]
            st.rerun()


def render_decision_tab(result: Result) -> None:
    rows = decision_rows(result)
    left, right = st.columns([3, 2])
    with left:
        st.altair_chart(waterfall_chart(rows), width="stretch")
    with right:
        st.markdown(layer_interval_html(result), unsafe_allow_html=True)
    st.markdown("#### Sectors")
    c1, c2, _ = st.columns([2, 2, 4])
    with c1:
        sort_by = st.selectbox("Sort by", list(SECTOR_COLUMNS), index=list(SECTOR_COLUMNS).index("Total"))
    with c2:
        order = st.radio("Order", ["Descending", "Ascending"], horizontal=True)
    st.markdown(sector_table_html(result, sort_by, order == "Descending"), unsafe_allow_html=True)


def render_factor_tab(result: Result) -> None:
    st.markdown(factor_table_html(result), unsafe_allow_html=True)
    st.markdown("#### Rolling exposures")
    available = list(result.rolling.exposures) if result.rolling is not None else []
    default = [f for f in ("market", "size") if f in available]
    chosen = st.multiselect("Factors", available, default=default, format_func=lambda f: f.replace("_", " "))
    if chosen:
        st.altair_chart(rolling_chart(result, chosen), width="stretch")
    st.markdown('<p class="note">The rolling path is for charting only; it is never used in any total.</p>',
                unsafe_allow_html=True)
    with st.expander("Diagnostics", expanded=False):
        st.dataframe(diagnostics_frame(result), hide_index=True, width="stretch")


def main() -> None:
    st.set_page_config(page_title="Return attribution", layout="wide")
    st.markdown(_CSS, unsafe_allow_html=True)
    try:
        loaded = _current()
    except (FileNotFoundError, ValueError) as exc:
        st.error(f"No result could be loaded: {exc}")
        return
    result = loaded.result
    st.markdown(header_html(loaded), unsafe_allow_html=True)
    render_controls(loaded)
    st.markdown(headline_html(result), unsafe_allow_html=True)
    decision_tab, factor_tab = st.tabs(["Decision view", "Factor view"])
    with decision_tab:
        render_decision_tab(result)
    with factor_tab:
        render_factor_tab(result)
    st.markdown(reconciliation_html(result), unsafe_allow_html=True)
    st.markdown(limitations_html(result), unsafe_allow_html=True)


if __name__ == "__main__":
    main()
