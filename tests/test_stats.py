from __future__ import annotations

import numpy as np
import pytest

from attribution.stats import DEFAULT_BLOCK_LENGTH, DEFAULT_ITERATIONS, block_bootstrap


def test_defaults_match_contract() -> None:
    assert DEFAULT_ITERATIONS == 2000
    assert DEFAULT_BLOCK_LENGTH == {"monthly": 6, "daily": 60}
    r = block_bootstrap(np.random.default_rng(1).normal(0.01, 0.02, 60))
    assert r.n_iterations == 2000 and r.block_length == 6 and r.level == 0.95
    d = block_bootstrap(np.random.default_rng(1).normal(0, 0.01, 500), frequency="daily")
    assert d.block_length == 60


def test_reproducible_with_seed() -> None:
    x = np.random.default_rng(2).normal(0.0, 1.0, 60)
    assert block_bootstrap(x, seed=5) == block_bootstrap(x, seed=5)
    assert block_bootstrap(x, seed=5) != block_bootstrap(x, seed=6)


def test_interval_brackets_point_and_flags_zero() -> None:
    noise = np.random.default_rng(3).normal(0.0, 0.05, 60)
    r = block_bootstrap(noise)
    assert r.lower <= r.point <= r.upper
    assert r.spans_zero
    assert "SPANS ZERO" in r.describe()


def test_clearly_positive_series_does_not_span_zero() -> None:
    x = 0.01 + np.random.default_rng(4).normal(0.0, 0.001, 60)
    r = block_bootstrap(x, lambda a: float(a.mean()) * 12)
    assert r.lower > 0 and not r.spans_zero
    assert "SPANS ZERO" not in r.describe()
    assert r.point == pytest.approx(x.mean() * 12)


def test_block_longer_than_series_is_capped() -> None:
    assert block_bootstrap(np.arange(10.0), frequency="daily").block_length == 10


def test_converts_to_result_confidence_interval() -> None:
    ci = block_bootstrap(np.random.default_rng(5).normal(0, 1, 30)).to_confidence_interval()
    assert ci.level == 0.95 and ci.lower <= ci.upper and "bootstrap" in ci.method


@pytest.mark.parametrize("bad", [[1.0], [1.0, float("nan")], np.ones((3, 3))])
def test_bad_input_rejected(bad: object) -> None:
    with pytest.raises(ValueError):
        block_bootstrap(bad)  # type: ignore[arg-type]
