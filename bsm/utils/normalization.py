from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jaxtyping import PyTree


class Data(NamedTuple):
    inputs: chex.Array
    outputs: chex.Array


class Stats(NamedTuple):
    mean: chex.Array
    std: chex.Array


class DataStats(NamedTuple):
    inputs: Stats
    outputs: Stats


class Normalizer:
    def __init__(self, num_correction=1e-6):
        self.num_correction = num_correction

    def compute_stats(self, data: PyTree) -> PyTree[Stats]:
        return jtu.tree_map(self.get_stats, data)

    def init_stats(self, data: PyTree) -> PyTree[Stats]:
        return jtu.tree_map(self._init_stats, data)

    @partial(jax.jit, static_argnums=0)
    def get_stats(self, data: chex.Array) -> Stats:
        assert data.ndim == 2
        mean = jnp.mean(data, axis=0)
        if data.shape[0] > 1:
            std = jnp.std(data, axis=0) + self.num_correction
        else:
            std = jnp.ones_like(mean)
        return Stats(mean, std)

    @partial(jax.jit, static_argnums=0)
    def _init_stats(self, data: chex.Array) -> Stats:
        assert data.ndim == 2
        mean = jnp.mean(data, axis=0)
        return Stats(jnp.zeros_like(mean), jnp.ones_like(mean))

    @partial(jax.jit, static_argnums=0)
    def normalize(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return (datum - stats.mean) / stats.std

    def _normalize_y(
        self, y: jnp.ndarray, stats: Stats, eps: float = 1e-8
    ) -> jnp.ndarray:
        y_normalized = (y - stats.mean) / (stats.std + eps)
        assert y_normalized.shape == y.shape
        return y_normalized

    @partial(jax.jit, static_argnums=0)
    def normalize_std(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return datum / stats.std

    @partial(jax.jit, static_argnums=0)
    def denormalize(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return datum * stats.std + stats.mean

    @partial(jax.jit, static_argnums=0)
    def denormalize_std(self, datum: chex.Array, stats: Stats) -> chex.Array:
        assert datum.ndim == 1
        return datum * stats.std
