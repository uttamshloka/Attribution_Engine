"""Command line: ``python -m attribution run --portfolio-name ... [--data-dir ...] [--out-dir ...]``

Folders default to :mod:`attribution.settings` (environment variables, then
repository-relative defaults).

Writes ``attribution_<portfolio>_<asof>.json`` (asof = last month-end of the
period) and a run log ``attribution_<portfolio>_<asof>.log`` beside it. An
existing file is never overwritten: a second run for the same date writes
``..._<asof>_2.json``, then ``_3`` and so on. Files are created exclusively,
and only after the result has passed the validation gate, so a failed run
writes nothing and the last good file stays in place.
"""

from __future__ import annotations

import argparse
import re
import sys
from collections.abc import Sequence
from pathlib import Path

from attribution.engine import EngineConfig, EngineError, run_attribution
from attribution.outputs import Result
from attribution.settings import data_dir, output_dir
from attribution.validate import GateError, validate_result

__all__ = ["safe_name", "write_result", "main"]


def safe_name(name: str) -> str:
    """Portfolio name made safe for a file name."""
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    if not cleaned:
        raise ValueError(f"portfolio name {name!r} has no usable characters")
    return cleaned


def _log_text(result: Result, json_name: str) -> str:
    r = result.run
    rec = result.reconciliation
    lines = [
        f"output            {json_name}",
        f"generated (UTC)   {r.generated_at_utc.isoformat()}",
        f"package version   {r.package_version}",
        f"linking method    {r.linking}",
        f"brinson variant   {r.brinson_variant}",
        f"factor order      {' -> '.join(r.factor_order)}",
        f"data folder       {r.data_dir}",
        "inputs:",
        *[f"  {k:18s} {v}" for k, v in r.inputs.items()],
        f"period            {result.attribution.period.start} to {result.attribution.period.end} "
        f"({result.attribution.period.n_periods} months)",
        f"validation        PASSED (decision - active {rec.decision_minus_active:.2e}, "
        f"factor - active {rec.factor_minus_active:.2e}, tolerance {rec.tolerance:g})",
        f"limitations       {len(result.limitations)} "
        f"({len(result.limitations_by_severity('material'))} material, "
        f"{len(result.limitations_by_severity('caution'))} caution, {len(result.limitations_by_severity('info'))} info)",
    ]
    return "\n".join(lines) + "\n"


def write_result(result: Result, out_dir: str | Path, portfolio_name: str) -> tuple[Path, Path]:
    """Validate, then write the JSON and its log without overwriting anything. Returns both paths."""
    validate_result(result)  # raises before anything touches the disk
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"attribution_{safe_name(portfolio_name)}_{result.attribution.period.end.isoformat()}"
    text = result.to_json()
    counter = 1
    while True:
        name = stem if counter == 1 else f"{stem}_{counter}"
        json_path, log_path = out / f"{name}.json", out / f"{name}.log"
        if not json_path.exists() and not log_path.exists():
            try:
                with open(json_path, "x", encoding="utf-8") as fh:
                    fh.write(text)
            except FileExistsError:
                counter += 1
                continue
            with open(log_path, "x", encoding="utf-8") as fh:
                fh.write(_log_text(result, json_path.name))
            return json_path, log_path
        counter += 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m attribution", description="Portfolio return attribution engine.")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run the engine on a data folder and write a validated JSON")
    run.add_argument("--data-dir", type=Path, default=None,
                     help="input folder (default: ATTRIBUTION_DATA_DIR or <repo>/sample_data)")
    run.add_argument("--out-dir", type=Path, default=None,
                     help="output folder (default: ATTRIBUTION_OUTPUT_DIR or a temp folder)")
    run.add_argument("--portfolio-name", required=True)
    run.add_argument("--policy", type=Path, default=None, help="policy JSON (default: <data-dir>/policy.json)")
    run.add_argument("--start", default=None, help="first month, e.g. 2022-01")
    run.add_argument("--end", default=None, help="last month, e.g. 2026-09")
    run.add_argument("--linking", choices=("carino", "menchero"), default="carino")
    run.add_argument("--no-bootstrap", action="store_true", help="skip confidence intervals (faster)")
    args = parser.parse_args(argv)

    if (args.start is None) != (args.end is None):
        parser.error("give both --start and --end, or neither")
    period = (args.start, args.end) if args.start else None
    cfg = EngineConfig(portfolio_name=args.portfolio_name, linking=args.linking, bootstrap=not args.no_bootstrap)
    data = args.data_dir or data_dir()
    out = args.out_dir or output_dir()
    try:
        result = run_attribution(data, args.policy, period=period, config=cfg)
        json_path, log_path = write_result(result, out, args.portfolio_name)
    except GateError as exc:
        print(str(exc), file=sys.stderr)
        print("Nothing was written; any earlier result file is unchanged.", file=sys.stderr)
        return 2
    except (EngineError, FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("Nothing was written.", file=sys.stderr)
        return 1
    print(f"wrote {json_path}")
    print(f"wrote {log_path}")
    return 0
