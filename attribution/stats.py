"""Block-bootstrap confidence intervals.

Returns are autocorrelated (a crisis is several bad months in a row), so
resampling single observations understates uncertainty. A block bootstrap
resamples contiguous blocks instead, keeping short-run dependence intact.

Method: circular moving-block bootstrap. Each replicate is built by drawing
block start points uniformly, taking ``block_length`` consecutive observations
from each (wrapping round the end of the series), concatenating, and trimming
to the original length. The statistic is computed on every replicate and the
interval is read off the percentiles of those values.

Defaults: 2,000 iterations; 6-observation blocks for monthly data and 60 for
daily data; 95% interval (2.5 / 97.5 percentiles); fixed seed, so results are
reproducible.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from attribution.result import ConfidenceInterval

__all__ = [
    "DEFAULT_BLOCK_LENGTH",
    "DEFAULT_ITERATIONS",
    "DEFAULT_SEED",
    "BootstrapResult",
    "block_bootstrap",
]

Frequency = Literal["monthly", "daily"]

DEFAULT_BLOCK_LENGTH: dict[str, int] = {"monthly": 6, "daily": 60}
DEFAULT_ITERATIONS = 2000
DEFAULT_SEED = 20261028
_METHOD = "circular moving-block bootstrap"


@dataclass(frozen=True)
class BootstrapResult:
    """Point estimate and percentile interval from a block bootstrap."""

    point: float
    lower: float
    upper: float
    level: float
    n_iterations: int
    block_length: int
    n_obs: int
    method: str = _METHOD

    @property
    def spans_zero(self) -> bool:
        """True when the interval includes zero: the estimate is not distinguishable from zero."""
        return self.lower <= 0.0 <= self.upper

    def describe(self, scale: float = 1.0, unit: str = "") -> str:
        pct_lo = (1 - self.level) / 2 * 100
        pct_hi = 100 - pct_lo
        text = (
            f"{self.point * scale:+.4g}{unit} "
            f"[{pct_lo:g}%: {self.lower * scale:+.4g}{unit}, {pct_hi:g}%: {self.upper * scale:+.4g}{unit}] "
            f"({self.method}, {self.n_iterations} iterations, block {self.block_length}, n={self.n_obs})"
        )
        if self.spans_zero:
            text += "  ** INTERVAL SPANS ZERO: not statistically distinguishable from zero **"
        return text

    def to_confidence_interval(self) -> ConfidenceInterval:
        return ConfidenceInterval(level=self.level, lower=self.lower, upper=self.upper, method=self.method)


def block_bootstrap(
    values: Sequence[float] | np.ndarray | pd.Series,
    statistic: Callable[[np.ndarray], float] = np.mean,
    *,
    frequency: Frequency = "monthly",
    block_length: int | None = None,
    n_iterations: int = DEFAULT_ITERATIONS,
    level: float = 0.95,
    seed: int = DEFAULT_SEED,
) -> BootstrapResult:
    """Bootstrap ``statistic`` over ``values`` and return the point estimate and interval.

    ``statistic`` receives a 1-D float array and returns a number; for example
    ``lambda a: a.mean() * 12`` for an annualised monthly mean. ``frequency``
    picks the default block length (6 monthly, 60 daily); ``block_length``
    overrides it. A block longer than the series is shortened to the series
    length.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 1:
        raise ValueError("values must be one-dimensional")
    n = x.size
    if n < 2:
        raise ValueError(f"need at least 2 observations to bootstrap, got {n}")
    if not np.all(np.isfinite(x)):
        raise ValueError("values must be finite (no NaN or infinity)")
    if frequency not in DEFAULT_BLOCK_LENGTH:
        raise ValueError(f"frequency must be one of {sorted(DEFAULT_BLOCK_LENGTH)}, got {frequency!r}")
    if not 0.0 < level < 1.0:
        raise ValueError(f"level must be in (0, 1), got {level}")
    if n_iterations < 100:
        raise ValueError("n_iterations must be at least 100")
    b = DEFAULT_BLOCK_LENGTH[frequency] if block_length is None else int(block_length)
    if b < 1:
        raise ValueError("block_length must be at least 1")
    b = min(b, n)

    rng = np.random.default_rng(seed)
    n_blocks = math.ceil(n / b)
    starts = rng.integers(0, n, size=(n_iterations, n_blocks))
    idx = ((starts[:, :, None] + np.arange(b)) % n).reshape(n_iterations, -1)[:, :n]
    samples = x[idx]
    stats = np.array([float(statistic(row)) for row in samples])
    if not np.all(np.isfinite(stats)):
        raise ValueError("statistic returned a non-finite value on some replicate")

    tail = (1.0 - level) / 2.0 * 100.0
    lower, upper = np.percentile(stats, [tail, 100.0 - tail])
    return BootstrapResult(
        point=float(statistic(x)),
        lower=float(lower),
        upper=float(upper),
        level=level,
        n_iterations=n_iterations,
        block_length=b,
        n_obs=n,
    )
