"""Fixed IPReservoir nodes that work correctly inside Model pipelines.

The library's IPReservoir extends TrainableNode, which causes two bugs when
used in a Model (e.g. ``Input >> IPReservoir >> Ridge``):

1. ``map_teacher`` assigns the teacher signal ``y`` to ALL TrainableNodes,
   but IPReservoir is unsupervised and doesn't need ``y``.  The dimension
   check (``y.dim == node.output_dim``) then fails because ``y`` has the
   output dimension while IPReservoir's ``output_dim`` equals its unit count.

2. ``model.fit`` never stores the output of unsupervised TrainableNodes for
   downstream consumption, so the next node (e.g. Ridge) receives an empty
   list instead of reservoir states.

These fixed versions extend the plain ``Node`` base class so the Model treats
them as regular (non-trainable) nodes.  Intrinsic-plasticity training happens
automatically on the first ``run()`` call, or can be triggered explicitly via
``train_ip()``.
"""

from functools import partial
from typing import Callable, Literal, Optional, Sequence, Union

from reservoirpy.type import (
    NodeInput,
    State,
    Timeseries,
    Timestep,
    Weights,
    is_array,
    is_multiseries,
)
from reservoirpy.utils.data_validation import check_node_input
from reservoirpy.utils.random import rand_generator
import jax
import jax.numpy as jnp
from numpy.random import Generator

from reservoirpy.jax.activationsfunc import get_function as get_function_jax
from reservoirpy.jax.mat_gen import bernoulli as bernoulli_jax, uniform as uniform_jax
from reservoirpy.jax.node import Node as NodeJAX
from reservoirpy.jax.utils import rand_generator as rand_generator_jax


class IPReservoir(NodeJAX):
    """Drop-in replacement for ``reservoirpy.jax.nodes.IPReservoir`` that
    works correctly inside a JAX ``Model`` pipeline.

    Intrinsic-plasticity training is performed automatically the first time
    ``run()`` is called (using the ``warmup`` value passed to the constructor).
    You can also call ``train_ip(x, warmup)`` explicitly.

    All constructor parameters are identical to the original ``IPReservoir``.
    """

    W: Weights
    Win: Weights
    bias: Weights
    a: float
    b: float
    lr: Union[float, jax.Array]
    sr: float
    mu: float
    sigma: float
    learning_rate: float
    epochs: int
    input_scaling: Union[float, Sequence]
    rc_connectivity: float
    input_connectivity: float
    activation: Literal["tanh", "sigmoid"]
    units: int
    rng: Generator

    def __init__(
        self,
        units: Optional[int] = None,
        sr: Optional[float] = None,
        lr: Union[float, jax.Array] = 1.0,
        mu: float = 0.0,
        sigma: float = 1.0,
        learning_rate: float = 5e-4,
        epochs: int = 1,
        input_scaling: Union[float, Sequence] = 1.0,
        input_connectivity: float = 0.1,
        rc_connectivity: float = 0.1,
        Win: Union[Weights, Callable] = bernoulli_jax,
        W: Union[Weights, Callable] = uniform_jax,
        bias: Union[Weights, Callable] = bernoulli_jax,
        activation: Literal["tanh", "sigmoid"] = "tanh",
        input_dim: Optional[int] = None,
        dtype: type = jnp.float64,
        seed: Optional[Union[int, Generator]] = None,
        warmup: int = 0,
        name=None,
    ):
        self.units = units
        self.sr = sr
        self.lr = lr
        self.mu = mu
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.input_scaling = input_scaling
        self.input_connectivity = input_connectivity
        self.rc_connectivity = rc_connectivity
        self.Win = Win
        self.W = W
        self.bias = bias
        self.activation = get_function_jax(activation)

        if activation == "tanh":
            self.gradient = partial(
                IPReservoir.gaussian_gradients, mu=mu, sigma=sigma
            )
        elif activation == "sigmoid":
            self.gradient = partial(IPReservoir.exp_gradients, mu=mu)
        else:
            raise ValueError(
                f"Activation '{activation}' must be 'tanh' or 'sigmoid' when "
                "applying intrinsic plasticity."
            )

        # set input_dim
        if input_dim is not None and is_array(Win) and Win.shape[-1] != input_dim:
            raise ValueError(
                f"Both 'input_dim' and 'Win' are set but their dimensions "
                f"don't match: {input_dim} != {Win.shape[-1]}."
            )
        self.input_dim = input_dim
        if is_array(Win):
            self.input_dim = jnp.shape(Win)[-1]

        # set output_dim
        if units is None and not is_array(W):
            raise ValueError(
                "'units' parameter must not be None if 'W' parameter is not a matrix."
            )
        if units is not None and is_array(W) and W.shape[-1] != units:
            raise ValueError(
                f"Both 'units' and 'W' are set but their dimensions don't "
                f"match: {units} != {W.shape[-1]}."
            )
        self.output_dim = units
        if is_array(W):
            self.output_dim = W.shape[-1]
            self.units = W.shape[-1]

        self.dtype = dtype
        self.rng = rand_generator_jax(seed=seed)
        self.name = name

        # IP training state
        self._ip_trained = False
        self._ip_warmup = warmup

    # -- initialisation -----------------------------------------------------

    def initialize(self, x, y=None):
        self._set_input_dim(x)

        Win_rng, W_rng, bias_rng = self.rng.spawn(3)

        if callable(self.Win):
            self.Win = self.Win(
                self.units,
                self.input_dim,
                input_scaling=self.input_scaling,
                connectivity=self.input_connectivity,
                dtype=self.dtype,
                seed=Win_rng,
            )

        if callable(self.W):
            self.W = self.W(
                self.units,
                self.units,
                sr=self.sr,
                connectivity=self.rc_connectivity,
                dtype=self.dtype,
                seed=W_rng,
            )

        if callable(self.bias):
            self.bias = self.bias(
                self.units,
                connectivity=1.0,
                dtype=self.dtype,
                seed=bias_rng,
            )

        self.a = jnp.ones((self.output_dim,))
        self.b = jnp.zeros((self.output_dim,))

        self.state = dict(
            internal=jnp.zeros((self.output_dim,)),
            out=jnp.zeros((self.output_dim,)),
        )
        self.initialized = True

    # -- forward pass -------------------------------------------------------

    @partial(jax.jit, static_argnums=(0,))
    def _step(self, state: State, x: Timestep) -> State:
        W = self.W
        Win = self.Win
        bias = self.bias
        f = self.activation
        lr = self.lr
        internal, external = state["internal"], state["out"]

        next_state = W @ external + Win @ x + bias
        next_state = (1 - lr) * internal + lr * next_state
        next_external = f(self.a * next_state + self.b)

        return {"internal": next_state, "out": next_external}

    # -- intrinsic-plasticity training --------------------------------------

    def train_ip(self, x: NodeInput, warmup: int = 0) -> "IPReservoirJAXFixed":
        """Train intrinsic-plasticity parameters *a* and *b*.

        Parameters
        ----------
        x : array-like of shape ([series,] timesteps, input_dim)
            Input data.
        warmup : int, default 0
            Number of initial timesteps to discard before training.

        Returns
        -------
        IPReservoirJAXFixed
            The node with updated IP parameters.
        """
        check_node_input(x, expected_dim=self.input_dim)

        if not self.initialized:
            self.initialize(x)

        for _epoch in range(self.epochs):
            if is_multiseries(x):
                for seq in x:
                    # Run warmup to advance state before IP training
                    if warmup > 0:
                        for u in seq[:warmup]:
                            self.step(u)
                    self._partial_fit_ip(seq[warmup:])
            else:
                # Run warmup to advance state before IP training
                if warmup > 0:
                    for u in x[:warmup]:
                        self.step(u)
                self._partial_fit_ip(x[warmup:])

        self._ip_trained = True
        return self

    def _partial_fit_ip(self, x: Timeseries):
        for u in x:
            post_state = self.step(u)
            pre_state = self.state["internal"]

            delta_a, delta_b = self.gradient(
                x=pre_state.T, y=post_state.T, a=self.a
            )
            self.a += self.learning_rate * delta_a
            self.b += self.learning_rate * delta_b

    # -- run (with auto IP training) ----------------------------------------

    def run(self, x=None, iters=None, workers=1):
        """Run the reservoir.  If IP has not been trained yet, it is trained
        automatically before producing output."""
        if not self._ip_trained and x is not None:
            self.train_ip(x, warmup=self._ip_warmup)
            self.reset()
        return super().run(x, iters=iters, workers=workers)

    # -- gradient helpers ---------------------------------------------------

    @staticmethod
    def gaussian_gradients(x, y, a, mu, sigma):
        sig2 = sigma ** 2
        delta_b = -(-(mu / sig2) + (y / sig2) * (2 * sig2 + 1 - y ** 2 + mu * y))
        delta_a = (1 / a) + delta_b * x
        return delta_a, delta_b

    @staticmethod
    def exp_gradients(x, y, a, mu):
        delta_b = 1 - (2 + (1 / mu)) * y + (y ** 2) / mu
        delta_a = (1 / a) + delta_b * x
        return delta_a, delta_b
