from numpy.typing import NDArray
from numpy import float64, ndarray, object_
from numpy.random import seed
from typing import Any
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import pytest
from psytest import PSYBubbles
from psytest.utils.functions import simulate_markov, r0_default
from psytest.utils.defaults import LAGMAX


@pytest.fixture(scope="session")
def sim_data() -> NDArray[float64]:
    seed(19210201)
    data: NDArray[float64] = simulate_markov(1000, 0.975, [1.01, 0.99])[1]
    return data


@pytest.fixture(scope="session")
def psy_params(sim_data: NDArray[float64]) -> dict[str, Any]:
    return {
        "data": sim_data,
        "r0": 0.10,
        "rstep": 0.10,
        "minlength": 0.10,
        "kmax": LAGMAX,
    }


@pytest.fixture(scope="session")
def psy_instance(psy_params: dict[str, Any]) -> PSYBubbles:
    return PSYBubbles(**psy_params)


@pytest.fixture(scope="session")
def simulation_params(psy_instance: PSYBubbles) -> dict[str, Any]:
    return {
        "nreps": 100,
        "nobs": 100,
    }


def test_init_all_args(psy_instance: PSYBubbles, sim_data: NDArray[float64]) -> None:
    assert isinstance(psy_instance, PSYBubbles)
    assert (psy_instance.data == sim_data).all()
    assert psy_instance.rstep == 0.10
    assert psy_instance.minlength == 0.10
    assert psy_instance.nobs == 1000
    assert isinstance(psy_instance.r0, float)
    assert isinstance(psy_instance.data, ndarray)
    assert psy_instance.data.dtype == float64
    assert len(psy_instance.data) == 1000
    assert psy_instance.lagmax == LAGMAX


def test_init_with_defaults(psy_params: dict[str, Any]) -> None:
    # Remove defaulted parameters
    psy_params.pop("r0")
    psy_params.pop("rstep")
    psy_params.pop("kmax")
    # Create instance
    psy_instance = PSYBubbles(**psy_params)
    assert isinstance(psy_instance, PSYBubbles)
    assert psy_instance.r0 == r0_default(psy_instance.nobs)
    assert psy_instance.rstep == 1 / psy_instance.nobs
    assert psy_instance.lagmax == LAGMAX


def test_teststat(psy_instance: PSYBubbles) -> None:
    test_stat: dict[float, float] = psy_instance.teststat()
    for r, ts in test_stat.items():
        assert isinstance(r, float)
        assert isinstance(ts, float)


def test_critval_tabulated(psy_instance: PSYBubbles) -> None:
    critval = psy_instance.critval(alpha=0.05, fast=True)
    for r, cv in critval.items():
        assert isinstance(r, float)
        assert isinstance(cv, float)


def test_critval_simulated(
    psy_instance: PSYBubbles, simulation_params: dict[str, Any]
) -> None:
    critval = psy_instance.critval(alpha=0.05, fast=False, **simulation_params)
    assert isinstance(critval, dict)
    for r, cv in critval.items():
        assert isinstance(r, float)
        assert isinstance(cv, float)


def test_critval_simulated_cache(
    psy_instance: PSYBubbles, simulation_params: dict[str, Any]
) -> None:
    # Run it first time
    critval: dict[float, NDArray[float64]] = psy_instance.critval(
        alpha=0.05, fast=False, **simulation_params
    )
    # Run it after caching
    critval2: dict[float, NDArray[float64]] = psy_instance.critval(
        alpha=0.05, fast=False, **simulation_params
    )
    assert critval == critval2
    assert hasattr(psy_instance.critval, "cache_info")
    cache = psy_instance.critval.cache_info()
    assert cache.hits > 0


def test_bubbles(psy_instance: PSYBubbles) -> None:
    bubbles: NDArray[object_] = psy_instance.find_bubbles(alpha=0.05, fast=True)
    assert isinstance(bubbles, ndarray)
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
