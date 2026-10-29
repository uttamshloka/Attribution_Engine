"""Where the package reads and writes. This is the ONLY place paths are decided.

Every location defaults to a folder relative to the repository (the folder
that contains the ``attribution`` package and ``streamlit_app.py``), and each
can be overridden by an environment variable:

=========================  ===================================  ======================================
setting                    environment variable                 default
=========================  ===================================  ======================================
``data_dir()``             ``ATTRIBUTION_DATA_DIR``             ``<repo>/sample_data``
``results_dir()``          ``ATTRIBUTION_RESULTS_DIR``          ``<repo>/results`` (committed, read-only)
``output_dir()``           ``ATTRIBUTION_OUTPUT_DIR``           ``<system temp>/attribution_outputs``
=========================  ===================================  ======================================

Streamlit Community Cloud has no persistent disk, so the app only READS from
``data_dir`` and ``results_dir`` (both committed to the repository) and only
WRITES to ``output_dir``, a temporary folder that may vanish on restart.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

__all__ = ["REPO_ROOT", "ENV_DATA_DIR", "ENV_RESULTS_DIR", "ENV_OUTPUT_DIR", "data_dir", "results_dir", "output_dir"]

REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_DATA_DIR = "ATTRIBUTION_DATA_DIR"
ENV_RESULTS_DIR = "ATTRIBUTION_RESULTS_DIR"
ENV_OUTPUT_DIR = "ATTRIBUTION_OUTPUT_DIR"


def _from_env(name: str, default: Path) -> Path:
    value = os.environ.get(name, "").strip()
    return Path(value).expanduser() if value else default


def data_dir() -> Path:
    """Input data folder (CSV/Parquet files and policy.json)."""
    return _from_env(ENV_DATA_DIR, REPO_ROOT / "sample_data")


def results_dir() -> Path:
    """Folder holding the committed, pre-computed result JSON the app shows by default."""
    return _from_env(ENV_RESULTS_DIR, REPO_ROOT / "results")


def output_dir() -> Path:
    """Writable folder for results produced at run time (temporary on Streamlit Cloud)."""
    return _from_env(ENV_OUTPUT_DIR, Path(tempfile.gettempdir()) / "attribution_outputs")
