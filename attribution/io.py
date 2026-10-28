"""Loaders and writers for the input contracts.

Loaders read CSV or Parquet (chosen by file extension), validate against the
table's schema and return a typed DataFrame. Errors name the file, column and
row: for CSV the physical line number in the file, for Parquet the 1-based row.

CSV files may begin with comment lines starting with ``#``; they are skipped
and do not upset line numbering. Parquet files carry their header comment in
the schema metadata under :data:`COMMENT_METADATA_KEY`.
"""

from __future__ import annotations

import io as _io
import json
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from attribution.schemas import (
    BENCHMARK,
    CASH_RETURN,
    FACTOR_RETURNS,
    HOLDINGS,
    PORTFOLIO_RETURN,
    PRICES,
    REGIME_SERIES,
    SECURITY_MASTER,
    Issue,
    Policy,
    SchemaValidationError,
    TableSchema,
)

__all__ = [
    "COMMENT_METADATA_KEY",
    "JSON_COMMENT_KEY",
    "SUPPORTED_SUFFIXES",
    "read_raw_table",
    "read_header_comment",
    "load_table",
    "write_table",
    "load_holdings",
    "load_security_master",
    "load_prices",
    "load_benchmark",
    "load_portfolio_returns",
    "load_factor_returns",
    "load_cash_returns",
    "load_regime_series",
    "load_policy",
    "write_policy",
]

COMMENT_METADATA_KEY = b"attribution.comment"
#: A JSON file may carry its header comment under this top-level key.
JSON_COMMENT_KEY = "_comment"
SUPPORTED_SUFFIXES = (".csv", ".parquet")


def _require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{path}: file not found")
    if not path.is_file():
        raise IsADirectoryError(f"{path}: not a regular file")


def _read_csv(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Return (frame of strings, row labels, comment lines)."""
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    comments: list[str] = []
    header_index = 0
    while header_index < len(lines) and lines[header_index].startswith("#"):
        comments.append(lines[header_index])
        header_index += 1
    if header_index >= len(lines) or not lines[header_index].strip():
        raise SchemaValidationError("CSV", str(path), [Issue(None, (), "no header row found")])
    body = "\n".join(lines[header_index:])
    try:
        frame = pd.read_csv(
            _io.StringIO(body),
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            skip_blank_lines=False,
        )
    except pd.errors.ParserError as exc:
        raise SchemaValidationError("CSV", str(path), [Issue(None, (), f"could not parse CSV: {exc}")]) from None
    # Physical line numbers are 1-based; the header sits on line header_index + 1.
    first_data_line = header_index + 2
    labels = [f"line {first_data_line + i}" for i in range(len(frame))]
    frame = frame.astype(object)
    return frame, labels, comments


def _read_parquet(path: Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    try:
        table = pq.read_table(path)
    except (pa.ArrowInvalid, OSError) as exc:
        raise SchemaValidationError("Parquet", str(path), [Issue(None, (), f"could not read Parquet: {exc}")]) from None
    frame = table.to_pandas()
    labels = [f"row {i + 1}" for i in range(len(frame))]
    meta = table.schema.metadata or {}
    comment = meta.get(COMMENT_METADATA_KEY)
    comments = comment.decode("utf-8").splitlines() if comment else []
    return frame, labels, comments


def read_raw_table(path: str | Path) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Read a CSV or Parquet file without validation.

    Returns the frame, a human row label per row, and any header comment lines.
    """
    p = Path(path)
    _require_file(p)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return _read_csv(p)
    if suffix == ".parquet":
        return _read_parquet(p)
    raise ValueError(f"{p}: unsupported file type {suffix!r}; expected one of {', '.join(SUPPORTED_SUFFIXES)}")


def read_header_comment(path: str | Path) -> list[str]:
    """Return the header comment lines of a CSV, Parquet or JSON file."""
    p = Path(path)
    _require_file(p)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8"))
        comment = data.get(JSON_COMMENT_KEY) if isinstance(data, dict) else None
        return comment.splitlines() if isinstance(comment, str) else []
    return read_raw_table(p)[2]


def load_table(path: str | Path, schema: TableSchema) -> pd.DataFrame:
    """Load ``path`` and validate it against ``schema``."""
    frame, labels, _ = read_raw_table(path)
    return schema.validate(frame, source=str(path), row_labels=labels)


def write_table(
    df: pd.DataFrame,
    path: str | Path,
    schema: TableSchema,
    header_comment: Sequence[str] = (),
) -> Path:
    """Validate ``df`` and write it as CSV or Parquet with an optional header comment.

    Dates are written as ``YYYY-MM-DD``. Comment lines are prefixed with ``# ``
    in CSV and stored as metadata in Parquet.
    """
    p = Path(path)
    clean = schema.validate(df, source=f"<write {p.name}>")
    suffix = p.suffix.lower()
    out = clean.copy()
    for spec in schema.columns:
        if spec.kind == "date":
            out[spec.name] = out[spec.name].dt.strftime("%Y-%m-%d")
    p.parent.mkdir(parents=True, exist_ok=True)
    if suffix == ".csv":
        buffer = _io.StringIO()
        for line in header_comment:
            buffer.write(f"# {line}\n")
        out.to_csv(buffer, index=False, lineterminator="\n", float_format="%.12g")
        p.write_text(buffer.getvalue(), encoding="utf-8")
    elif suffix == ".parquet":
        table = pa.Table.from_pandas(out, preserve_index=False)
        meta = dict(table.schema.metadata or {})
        meta.pop(b"pandas", None)
        if header_comment:
            meta[COMMENT_METADATA_KEY] = "\n".join(header_comment).encode("utf-8")
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, p)
    else:
        raise ValueError(f"{p}: unsupported file type {suffix!r}; expected one of {', '.join(SUPPORTED_SUFFIXES)}")
    return p


def load_holdings(path: str | Path) -> pd.DataFrame:
    return load_table(path, HOLDINGS)


def load_security_master(path: str | Path) -> pd.DataFrame:
    return load_table(path, SECURITY_MASTER)


def load_prices(path: str | Path) -> pd.DataFrame:
    return load_table(path, PRICES)


def load_benchmark(path: str | Path) -> pd.DataFrame:
    return load_table(path, BENCHMARK)


def load_portfolio_returns(path: str | Path) -> pd.DataFrame:
    return load_table(path, PORTFOLIO_RETURN)


def load_factor_returns(path: str | Path) -> pd.DataFrame:
    return load_table(path, FACTOR_RETURNS)


def load_cash_returns(path: str | Path) -> pd.DataFrame:
    return load_table(path, CASH_RETURN)


def load_regime_series(path: str | Path) -> pd.DataFrame:
    return load_table(path, REGIME_SERIES)


def load_policy(path: str | Path) -> Policy:
    """Load a Policy from JSON. A top-level ``_comment`` string is allowed and ignored."""
    p = Path(path)
    _require_file(p)
    if p.suffix.lower() != ".json":
        raise ValueError(f"{p}: policy must be a .json file")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SchemaValidationError(
            "Policy", str(p), [Issue(None, (f"line {exc.lineno}",), f"invalid JSON: {exc.msg}")]
        ) from None
    if isinstance(data, dict):
        comment = data.pop(JSON_COMMENT_KEY, None)
        if comment is not None and not isinstance(comment, str):
            raise SchemaValidationError("Policy", str(p), [Issue(JSON_COMMENT_KEY, (), "must be a string")])
    return Policy.from_dict(data, source=str(p))


def write_policy(policy: Policy, path: str | Path, header_comment: Sequence[str] = ()) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, object] = {}
    if header_comment:
        payload[JSON_COMMENT_KEY] = "\n".join(header_comment)
    payload.update(policy.to_dict())
    p.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return p
