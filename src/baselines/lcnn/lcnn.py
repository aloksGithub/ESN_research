"""
Locally Connected Neural Network (LCNN) for time series prediction.

Python reimplementation of the C++ LCNN from:
  https://github.com/FloopCZ/echo-state-networks

Key design: the reservoir is a 2D state matrix (H x W) with local
convolutional kernels and periodic (wrap-around) boundary conditions.
"""
import numpy as np
from sklearn.linear_model import Ridge, ElasticNet


def lcnn_step(state, reservoir_w):
    """Apply the local convolution step with periodic boundary.

    Args:
        state: (H, W) current state matrix
        reservoir_w: (H, W, KH, KW) per-neuron kernel weights

    Returns:
        (H, W) new state delta from recurrent connections
    """
    kh = reservoir_w.shape[2]
    kw = reservoir_w.shape[3]
    half_kh = kh // 2
    half_kw = kw // 2
    new_state = np.zeros_like(state)
    for i in range(kh):
        for j in range(kw):
            shift_h = -i + half_kh
            shift_w = -j + half_kw
            shifted = np.roll(np.roll(state, shift_h, axis=0), shift_w, axis=1)
            new_state += reservoir_w[:, :, i, j] * shifted
    return new_state


class LCNN:
    """Locally Connected Neural Network (echo state network variant).

    The reservoir is a 2D grid of neurons (state_height x state_width).
    Each neuron has a local receptive field (kernel_height x kernel_width)
    with periodic boundary conditions (toroidal topology).

    Args:
        state_height: Number of rows in the state matrix.
        state_width: Number of columns in the state matrix.
        kernel_height: Kernel height (must be odd).
        kernel_width: Kernel width (must be odd).
        topology: Reservoir topology ('lcnn', 'conv', or 'sparse').
        sigma_res: Std of reservoir weight distribution.
        mu_res: Mean of reservoir weight distribution.
        sigma_in: Std of input weight distribution.
        mu_in: Mean of input weight distribution.
        sigma_fb: Std of feedback weight distribution.
        mu_fb: Mean of feedback weight distribution.
        sigma_b: Std of bias distribution.
        mu_b: Mean of bias distribution.
        sparsity: Fraction of reservoir connections set to zero.
        in_fb_sparsity: Fraction of input/feedback connections zeroed.
        leakage: Leaky integration rate (1.0 = no leak).
        noise: Std of multiplicative noise on state delta.
        act_steepness: Scaling factor inside tanh activation.
        spectral_radius: If > 0, rescale reservoir weights to this radius.
        ridge: Ridge regularization for readout training.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        state_height=11,
        state_width=11,
        kernel_height=5,
        kernel_width=5,
        topology='lcnn',
        sigma_res=1.0,
        mu_res=0.0,
        sigma_in=1.0,
        mu_in=0.0,
        sigma_fb=0.0,
        mu_fb=0.0,
        sigma_b=0.0,
        mu_b=0.0,
        sparsity=0.0,
        in_fb_sparsity=0.0,
        leakage=1.0,
        noise=0.0,
        act_steepness=1.0,
        spectral_radius=0.0,
        ridge=1e-6,
        seed=None,
    ):
        if kernel_height % 2 == 0 or kernel_width % 2 == 0:
            raise ValueError("Kernel size must be odd.")

        self.state_height = state_height
        self.state_width = state_width
        self.kernel_height = kernel_height
        self.kernel_width = kernel_width
        self.topology = topology
        self.sigma_res = sigma_res
        self.mu_res = mu_res
        self.sigma_in = sigma_in
        self.mu_in = mu_in
        self.sigma_fb = sigma_fb
        self.mu_fb = mu_fb
        self.sigma_b = sigma_b
        self.mu_b = mu_b
        self.sparsity = sparsity
        self.in_fb_sparsity = in_fb_sparsity
        self.leakage = leakage
        self.noise = noise
        self.act_steepness = act_steepness
        self.spectral_radius = spectral_radius
        self.ridge = ridge
        self.rng = np.random.RandomState(seed)

        self.state_ = None
        self.reservoir_w_ = None
        self.reservoir_w_full_ = None
        self.input_w_ = None
        self.feedback_w_ = None
        self.bias_ = None
        self.output_w_ = None
        self.n_neurons_ = state_height * state_width

    def _init_weights(self, n_inputs, n_outputs):
        """Initialize all weight matrices."""
        rng = self.rng
        H, W = self.state_height, self.state_width
        KH, KW = self.kernel_height, self.kernel_width
        N = self.n_neurons_

        # --- Reservoir weights ---
        if self.topology == 'sparse':
            self.reservoir_w_full_ = (
                self.sigma_res * (rng.rand(N, N) * 2 - 1) + self.mu_res
            )
            mask = rng.rand(N, N) >= self.sparsity
            self.reservoir_w_full_ *= mask
            if self.spectral_radius > 0:
                eigvals = np.abs(np.linalg.eigvals(self.reservoir_w_full_))
                sr = np.max(eigvals)
                if sr > 0:
                    self.reservoir_w_full_ *= self.spectral_radius / sr
        elif self.topology == 'conv':
            kernel = self.sigma_res * (rng.rand(KH, KW) * 2 - 1) + self.mu_res
            self.reservoir_w_ = np.tile(kernel, (H, W, 1, 1))
            mask = rng.rand(H, W, KH, KW) >= self.sparsity
            self.reservoir_w_ *= mask
            if self.spectral_radius > 0:
                self._rescale_local_sr()
        elif self.topology == 'lcnn':
            self.reservoir_w_ = (
                self.sigma_res * (rng.rand(H, W, KH, KW) * 2 - 1) + self.mu_res
            )
            mask = rng.rand(H, W, KH, KW) >= self.sparsity
            self.reservoir_w_ *= mask
            if self.spectral_radius > 0:
                self._rescale_local_sr()
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        # --- Input weights ---
        self.input_w_ = (
            self.sigma_in * (rng.rand(H, W, n_inputs) * 2 - 1) + self.mu_in
        )
        if self.in_fb_sparsity > 0:
            self.input_w_ *= rng.rand(H, W, n_inputs) >= self.in_fb_sparsity

        # --- Feedback weights ---
        if self.sigma_fb != 0 or self.mu_fb != 0:
            self.feedback_w_ = (
                self.sigma_fb * (rng.rand(H, W, n_outputs) * 2 - 1) + self.mu_fb
            )
            if self.in_fb_sparsity > 0:
                self.feedback_w_ *= rng.rand(H, W, n_outputs) >= self.in_fb_sparsity
        else:
            self.feedback_w_ = None

        # --- Biases ---
        self.bias_ = self.sigma_b * rng.randn(H, W) + self.mu_b

        # --- State ---
        self.state_ = np.zeros((H, W))
        self.output_w_ = None
        self.last_output_ = None

    def _rescale_local_sr(self):
        """Rescale local reservoir weights to match a target spectral radius.

        Builds the equivalent dense matrix from the local kernels, computes
        its spectral radius, and rescales.
        """
        dense = self._local_to_dense()
        eigvals = np.abs(np.linalg.eigvals(dense))
        sr = np.max(eigvals) if len(eigvals) > 0 else 1.0
        if sr > 0:
            self.reservoir_w_ *= self.spectral_radius / sr

    def _local_to_dense(self):
        """Convert local kernel weights to equivalent dense matrix."""
        H, W = self.state_height, self.state_width
        KH, KW = self.kernel_height, self.kernel_width
        N = H * W
        dense = np.zeros((N, N))
        half_kh = KH // 2
        half_kw = KW // 2
        for r in range(H):
            for c in range(W):
                dst = r * W + c
                for ki in range(KH):
                    for kj in range(KW):
                        sr = (r - ki + half_kh) % H
                        sc = (c - kj + half_kw) % W
                        src = sr * W + sc
                        dense[dst, src] += self.reservoir_w_[r, c, ki, kj]
        return dense

    def _step(self, input_vec, feedback_vec=None):
        """Perform one reservoir update step.

        Args:
            input_vec: (n_inputs,) input at this timestep.
            feedback_vec: (n_outputs,) previous output for teacher forcing.
        """
        state_delta = np.zeros_like(self.state_)

        # Recurrent connections
        if self.reservoir_w_ is not None:
            state_delta += lcnn_step(self.state_, self.reservoir_w_)
        else:
            flat = self.state_.flatten()
            state_delta += (self.reservoir_w_full_ @ flat).reshape(self.state_.shape)

        # Input
        input_w_2d = self.input_w_.reshape(self.n_neurons_, -1)
        state_delta += (input_w_2d @ input_vec).reshape(self.state_.shape)

        # Feedback
        if self.feedback_w_ is not None and feedback_vec is not None:
            fb_w_2d = self.feedback_w_.reshape(self.n_neurons_, -1)
            state_delta += (fb_w_2d @ feedback_vec).reshape(self.state_.shape)

        # Noise
        if self.noise > 0:
            state_delta *= 1.0 + self.rng.randn(*self.state_.shape) * self.noise

        # Leakage + activation (standard ESN formulation)
        self.state_ = (
            (1.0 - self.leakage) * self.state_
            + self.leakage * np.tanh(self.act_steepness * state_delta + self.bias_)
        )

    def _compute_output(self):
        """Compute network output from current state."""
        if self.output_w_ is None:
            return None
        predictors = np.concatenate([self.state_.flatten(), [1.0]])
        return self.output_w_ @ predictors

    def run(self, inputs, outputs=None, washout=0, teacher_forcing=True):
        """Feed a sequence through the network, collecting states.

        Args:
            inputs: (T, n_inputs) input sequence.
            outputs: (T, n_outputs) target sequence for teacher forcing.
            washout: Number of initial steps to discard from collected states.
            teacher_forcing: Whether to use target outputs as feedback.

        Returns:
            states: (T - washout, n_neurons) collected flattened states.
            predictions: (T - washout, n_outputs) or None if output_w not set.
        """
        T = len(inputs)
        states = []
        predictions = []

        for t in range(T):
            feedback = None
            if teacher_forcing and outputs is not None and self.last_output_ is not None:
                feedback = self.last_output_
            elif not teacher_forcing and self.last_output_ is not None and self.feedback_w_ is not None:
                feedback = self.last_output_

            self._step(inputs[t], feedback)

            if teacher_forcing and outputs is not None:
                self.last_output_ = outputs[t]
            else:
                pred = self._compute_output()
                if pred is not None:
                    self.last_output_ = pred

            if t >= washout:
                states.append(self.state_.flatten().copy())
                pred = self._compute_output()
                if pred is not None:
                    predictions.append(pred.copy())

        states = np.array(states)
        predictions = np.array(predictions) if predictions else None
        return states, predictions

    def train(self, states, targets, noise_augment=0.0):
        """Train the readout layer via ridge regression.

        Args:
            states: (T, n_neurons) collected reservoir states.
            targets: (T, n_outputs) target outputs.
            noise_augment: If > 0, augment training data with noisy copies
                of the states to improve AR robustness. The value is the std
                of additive Gaussian noise relative to each neuron's std.
        """
        if noise_augment > 0:
            n_copies = 3
            state_std = states.std(axis=0, keepdims=True)
            state_std = np.maximum(state_std, 1e-10)
            aug_states = [states]
            aug_targets = [targets]
            for _ in range(n_copies):
                noise = self.rng.randn(*states.shape) * state_std * noise_augment
                aug_states.append(states + noise)
                aug_targets.append(targets)
            states = np.vstack(aug_states)
            targets = np.vstack(aug_targets)

        predictors = np.hstack([states, np.ones((len(states), 1))])

        reg = Ridge(alpha=self.ridge, fit_intercept=False)
        reg.fit(predictors, targets)
        self.output_w_ = np.atleast_2d(reg.coef_)

    def fit(self, train_in, train_out, val_in=None, val_out=None,
            washout=100, noise_augment=0.0):
        """Full train pipeline: init weights, run with teacher forcing, train readout.

        Args:
            train_in: (T_train, n_inputs) training inputs.
            train_out: (T_train, n_outputs) training targets.
            val_in: Validation inputs (unused, for interface compat).
            val_out: Validation outputs (unused, for interface compat).
            washout: Steps to discard at the start.
            noise_augment: Noise augmentation factor for AR robustness.

        Returns:
            self
        """
        n_inputs = train_in.shape[1]
        n_outputs = train_out.shape[1]
        self._init_weights(n_inputs, n_outputs)

        states, _ = self.run(train_in, train_out, washout=washout, teacher_forcing=True)
        targets = train_out[washout:]
        self.train(states, targets, noise_augment=noise_augment)
        return self

    def predict(self, inputs, teacher_forcing=False):
        """Run the trained network on new inputs.

        Args:
            inputs: (T, n_inputs) input sequence.
            teacher_forcing: If False, uses own predictions as feedback.

        Returns:
            predictions: (T, n_outputs)
        """
        _, predictions = self.run(inputs, teacher_forcing=False)
        return predictions

    def predict_autoregressive(self, initial_input, steps):
        """Generate a multi-step autoregressive forecast.

        Args:
            initial_input: (n_inputs,) the first input.
            steps: Number of steps to generate.

        Returns:
            predictions: (steps, n_outputs)
        """
        preds = []
        current = initial_input.copy()
        for _ in range(steps):
            self._step(current, self.last_output_)
            pred = self._compute_output()
            self.last_output_ = pred
            preds.append(pred.copy())
            current = pred
        return np.array(preds)
