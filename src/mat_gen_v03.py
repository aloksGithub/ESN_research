"""v0.3-compatible weight initialization for use with v0.4 reservoirpy nodes.

The v0.4 mat_gen module changed the internal `_random_sparse` algorithm:
- v0.3: generates full matrix, then zeros out entries by random mask
- v0.4: only generates non-zero entries and places them at random positions

Same seed produces completely different weight matrices between versions.
This module provides v0.3's initialization logic wrapped as v0.4
JaxInitializer objects, so they can be passed as Win/W/bias to v0.4
Reservoir and IPReservoir constructors.

Usage:
    from src.mat_gen_v03 import bernoulli, normal, uniform

    reservoir = Reservoir(100, Win=bernoulli, W=normal, ...)
"""

from functools import partial
from typing import Callable, Literal, Union

import numpy as np
from scipy import sparse, stats

from reservoirpy.type import global_dtype
from reservoirpy.utils.random import rand_generator
from reservoirpy.jax.mat_gen import JaxInitializer


def _get_rvs(dist: str, random_state: np.random.Generator, **kwargs) -> Callable:
    if dist == "custom_bernoulli":
        return _bernoulli_discrete_rvs(**kwargs, random_state=random_state)
    elif dist in dir(stats):
        distribution = getattr(stats, dist)
        return partial(distribution(**kwargs).rvs, random_state=random_state)
    else:
        raise ValueError(f"'{dist}' is not a valid distribution name.")


def _bernoulli_discrete_rvs(
    p=0.5, value: float = 1.0, random_state=None
) -> Callable:
    rg = rand_generator(random_state)

    def rvs(size: int = 1):
        return rg.choice([value, -value], p=[p, 1 - p], replace=True, size=size)

    return rvs


def _random_sparse_v03(
    *shape: int,
    dist: str,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed=None,
    degree=None,
    direction: Literal["in", "out"] = "out",
    **kwargs,
):
    """v0.3's _random_sparse: generate full matrix then zero out entries."""
    rg = rand_generator(seed)
    rvs = _get_rvs(dist, **kwargs, random_state=rg)

    if degree is not None:
        raise NotImplementedError("degree-based generation not ported")

    if 0 < connectivity > 1.0:
        raise ValueError("'connectivity' must be >0 and <1.")

    if connectivity >= 1.0 or len(shape) != 2:
        # v0.3 approach: generate ALL values, then mask
        matrix = rvs(size=shape).astype(dtype)
        if connectivity < 1.0:
            matrix[rg.random(shape) > connectivity] = 0.0
    else:
        # v0.3 approach: scipy.sparse.random
        matrix = sparse.random(
            shape[0],
            shape[1],
            density=connectivity,
            format=sparsity_type,
            random_state=rg,
            data_rvs=rvs,
            dtype=dtype,
        )

    if type(matrix) is np.matrix:
        matrix = np.asarray(matrix)

    return matrix


def _bernoulli_v03(
    *shape: int,
    p: float = 0.5,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed=None,
    degree=None,
    direction: Literal["in", "out"] = "out",
):
    return _random_sparse_v03(
        *shape,
        dist="custom_bernoulli",
        p=p,
        connectivity=connectivity,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
        degree=degree,
        direction=direction,
    )


def _normal_v03(
    *shape: int,
    loc: float = 0.0,
    scale: float = 1.0,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed=None,
    degree=None,
    direction: Literal["in", "out"] = "out",
):
    return _random_sparse_v03(
        *shape,
        dist="norm",
        loc=loc,
        scale=scale,
        connectivity=connectivity,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
        degree=degree,
        direction=direction,
    )


def _uniform_v03(
    *shape: int,
    low: float = -1.0,
    high: float = 1.0,
    connectivity: float = 1.0,
    dtype: np.dtype = global_dtype,
    sparsity_type: str = "csr",
    seed=None,
    degree=None,
    direction: Literal["in", "out"] = "out",
):
    return _random_sparse_v03(
        *shape,
        dist="uniform",
        loc=low,
        scale=high - low,
        connectivity=connectivity,
        dtype=dtype,
        sparsity_type=sparsity_type,
        seed=seed,
        degree=degree,
        direction=direction,
    )


# Export as JaxInitializer-wrapped callables compatible with v0.4 nodes
bernoulli = JaxInitializer(_bernoulli_v03)
normal = JaxInitializer(_normal_v03)
uniform = JaxInitializer(_uniform_v03)
