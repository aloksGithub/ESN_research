"""Repository NVAR correction for immutable JAX array updates."""
from functools import partial

import jax
import jax.numpy as jnp
from reservoirpy.jax.nodes import NVAR as LibraryNVAR


class NVAR(LibraryNVAR):
    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state, x):
        store = jnp.roll(state["store"], 1, axis=0).at[0].set(x)
        linear = store[::self.strides].ravel()
        nonlinear = jnp.prod(linear[self._monomial_idx], axis=1)
        return {"out": jnp.concatenate((linear, nonlinear)), "store": store}
