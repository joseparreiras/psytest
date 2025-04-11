from numpy.typing import NDArray
from numpy import float64
from numpy.random import seed
from typing import Any
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pytest
from psytest import PSYBubbles
from psytest.utils.functions import simulate_markov


@pytest.fixture(scope="module")
def sim_data() -> NDArray[float64]:
    seed(19210201)
    data: NDArray[float64] = simulate_markov(1000, 0.975, [1.01, 0.99])[1]
    return data


@pytest.fixture(scope="module")
def psy_instance(sim_data) -> PSYBubbles:
    return PSYBubbles(sim_data, rstep=0.10, minlength=0.10)


def test_teststat(psy_instance: PSYBubbles) -> None:
    test_stat: dict[float, float] = psy_instance.teststat()
    for r, ts in test_stat.items():
        assert isinstance(r, float)
        assert isinstance(ts, float)


def test_critval_tabulated(psy_instance: PSYBubbles) -> None:
    critval = psy_instance.critval(test_size=0.05, fast=True)
    for r, cv in critval.items():
        assert isinstance(r, float)
        assert isinstance(cv, float)


def test_critval_simulated(psy_instance: PSYBubbles) -> None:
    sim_kwargs: dict[str, Any] = {
        "r0": psy_instance.r0,
        "rstep": psy_instance.rstep,
        "nreps": 100,
        "nobs": 100,
    }
    critval = psy_instance.critval(test_size=0.05, fast=False, **sim_kwargs)
    assert isinstance(critval, dict)
    for r, cv in critval.items():
        assert isinstance(r, float)
        assert isinstance(cv, float)


def test_bubbles(psy_instance: PSYBubbles) -> None:
    bubbles = psy_instance.find_bubbles(alpha=0.05, fast=True)
    assert isinstance(bubbles, dict)
    for b in bubbles:
        assert isinstance(b, tuple)
        assert len(b) == 2
        start, end = b
        assert isinstance(start, int)
        assert 0 <= start <= psy_instance.nobs
        if end is not None:
            assert isinstance(end, int)
            assert 0 <= end <= psy_instance.nobs
            assert start < end
