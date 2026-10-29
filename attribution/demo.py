"""Build the committed demo: the synthetic sample data and its pre-computed result.

``python -m attribution.demo`` regenerates ``sample_data/`` and writes exactly
one result file to ``results/``, replacing the previous demo result. (This is
the one place a result file is replaced rather than numbered: ``results/`` holds
the single demo file the app loads, not an audit trail. Real runs go through
:func:`attribution.cli.write_result`, which never overwrites.)
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from attribution.engine import EngineConfig, run_attribution
from attribution.outputs import Result
from attribution.sample import generate_sample, write_sample
from attribution.settings import data_dir, results_dir
from attribution.validate import validate_result

__all__ = ["DEMO_PORTFOLIO", "build_demo", "main"]

DEMO_PORTFOLIO = "Sample PMS"


def build_demo(data: Path | None = None, results: Path | None = None) -> tuple[Path, Result]:
    """Regenerate the sample data, run the engine, and write the demo result. Returns (path, result)."""
    data = data or data_dir()
    results = results or results_dir()
    write_sample(generate_sample(), data)
    result = validate_result(run_attribution(data, config=EngineConfig(portfolio_name=DEMO_PORTFOLIO)))
    results.mkdir(parents=True, exist_ok=True)
    target = results / f"attribution_{DEMO_PORTFOLIO.replace(' ', '_')}_{result.attribution.period.end.isoformat()}.json"
    fd, tmp = tempfile.mkstemp(dir=results, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(result.to_json())
    os.chmod(tmp, 0o644)
    os.replace(tmp, target)  # atomic: a reader never sees a half-written file
    for old in results.glob("attribution_*.json"):
        if old != target:
            old.unlink()
    return target, result


def main() -> int:
    path, result = build_demo()
    print(f"sample data   {data_dir()}")
    print(f"demo result   {path}")
    print(f"period        {result.attribution.period.start} to {result.attribution.period.end}")
    print(f"validation    PASSED (net active {result.attribution.totals.active:+.4%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
