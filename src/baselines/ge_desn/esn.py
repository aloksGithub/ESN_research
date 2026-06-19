"""
Growing Evolutional Deep Echo State Network (GE-DESN).

Adapted from ESNGIPMAMG.py by Shen et al. (Neurocomputing 611, 2025).
Original source: https://github.com/ShenZeroQy/Growing-Evolutional-Deep-ESN

Changes from original:
  - Removed file-based data loading; data is passed in directly
  - Removed GPU dependency (uses CPU entropy only)
  - Removed PSO hyperparameter optimization
  - Removed Excel/plotting side effects
  - Removed `from numpy import *`; uses explicit np.* calls
  - Kept all core ESN logic (reservoir init, training, merging, pruning) intact
"""
import numpy as np
from . import entropy as Entropy


class EchoStateNetwork:
    """Deep Echo State Network with grow/evolve support.

    Args:
        U_init: Input data for washout, shape (input_dim, init_len)
        U_train: Input data for training, shape (input_dim, train_len)
        Y_train: Target data for training, shape (output_dim, train_len)
        U_val: Input data for validation, shape (input_dim, val_len)
        Y_val: Target data for validation, shape (output_dim, val_len)
        U_test: Input data for testing, shape (input_dim, test_len)
        Y_test: Target data for testing, shape (output_dim, test_len)
        pram: Dict of structural parameters
        oram: Dict of weight-scaling / regularization parameters
    """

    def __init__(self, U_init, U_train, Y_train, U_val, Y_val,
                 U_test, Y_test, pram, oram):
        self.U_dim = pram['input_dim']
        self.Y_dim = pram['output_dim']
        self.galaph = pram['leaky_rate']
        self.Stacklayer = pram['max_layers']
        self.InitX_dim = pram['neurons_add']

        self.ampWi = oram['ampWi']
        self.ampWc = oram['ampWp']
        self.ampWr = oram['ampWr']
        self.Reg_fac = oram['reg_fac']
        self.SpareRate = oram['spare_rate']

        self.Stack = 1
        self.X_dim = []

        # Store data references
        self.U_init = U_init
        self.U_train = U_train
        self.Y_train = Y_train
        self.U_val = U_val
        self.Y_val = Y_val
        self.U_test = U_test
        self.Y_test = Y_test
        self.TrainProcessLen = U_train.shape[1]

    # ------------------------------------------------------------------ #
    #  Reservoir dynamics                                                  #
    # ------------------------------------------------------------------ #

    def UspanX(self, uj, alaph):
        """Propagate one input timestep through all layers."""
        self.GroupX[0] = ((1 - alaph) * self.GroupX[0]
                          + alaph * np.tanh(self.GroupWin[0] @ uj
                                            + self.GroupW[0] @ self.GroupX[0]))
        for i in range(1, self.Stack):
            self.GroupX[i] = ((1 - alaph) * self.GroupX[i]
                              + alaph * np.tanh(self.GroupC[i - 1] @ self.GroupX[i - 1]
                                                + self.GroupW[i] @ self.GroupX[i]))
        U_X = self.GroupX[0]
        for i in range(1, self.Stack):
            U_X = np.concatenate((U_X, self.GroupX[i]), axis=0)
        return U_X

    # ------------------------------------------------------------------ #
    #  Reservoir initialization                                            #
    # ------------------------------------------------------------------ #

    def Inilize_First_reservoir(self, X_dim):
        """Initialize the first reservoir layer with X_dim neurons."""
        self.GroupX = [np.random.rand(X_dim, 1)]
        self.GroupWin = [np.random.rand(X_dim, self.U_dim) * self.ampWi - self.ampWi * 0.5]
        self.GroupC = []

        self.GroupW = []
        W = self._make_reservoir_matrix(X_dim)
        self.GroupW.append(W)

        self.X_dim = [X_dim]
        self.ExistNode = np.ones((X_dim,), dtype=np.int32)
        self.Stack = 1

    def Inilize_Stack_a_reservoir(self, X_dim):
        """Add a new reservoir layer on top with X_dim neurons."""
        self.GroupX.append(np.random.rand(X_dim, 1))

        ci = (np.random.rand(X_dim, self.X_dim[self.Stack - 1]) - 0.5) * self.ampWc
        self.GroupC.append(ci)

        W = self._make_reservoir_matrix(X_dim)
        self.GroupW.append(W)

        self.X_dim.append(X_dim)
        self.ExistNode = np.concatenate(
            (self.ExistNode, np.ones((X_dim,), dtype=np.int32)), axis=0)
        self.Stack += 1

    def _make_reservoir_matrix(self, X_dim):
        """Create a sparsified, spectrally-scaled reservoir weight matrix."""
        core = np.random.rand(X_dim, X_dim) - 0.5
        spare = self._sparelize(core, X_dim, X_dim, self.SpareRate)
        core = (np.random.rand(X_dim, X_dim) - 0.5) * spare
        e_vals, _ = np.linalg.eig(core)
        lamda = np.abs(e_vals).max()
        if lamda == 0:
            lamda = 1.0
        return core / lamda * self.ampWr

    # ------------------------------------------------------------------ #
    #  Training & inference                                                #
    # ------------------------------------------------------------------ #

    def Init_reservior(self, U_init, alaph=None):
        """Washout: drive the reservoir with init data to reach steady state."""
        if alaph is None:
            alaph = self.galaph
        initLen = U_init.shape[1]
        for j in range(initLen):
            self.UspanX(U_init[:, j:j + 1], alaph)
        # Save post-washout state for reinit
        self.InitialX = [self.GroupX[i].copy() for i in range(self.Stack)]

    def Reinit_reservoir(self):
        """Restore reservoir states to post-washout snapshot."""
        for i in range(self.Stack):
            self.GroupX[i] = self.InitialX[i].copy()

    def Train_reservoir(self, U_train, Y_train):
        """Collect states and solve for Wout via ridge regression."""
        alaph = self.galaph
        total_neurons = sum(self.X_dim)
        X_train = np.zeros((total_neurons, U_train.shape[1]))
        for i in range(U_train.shape[1]):
            X_train[:, i:i + 1] = self.UspanX(U_train[:, i:i + 1], alaph)

        self.PWout = np.linalg.inv(
            X_train @ X_train.T + self.Reg_fac * np.eye(X_train.shape[0]))
        self.Wout = (self.PWout @ (X_train @ Y_train.T)).T
        return X_train

    def Validate_test_data_constant(self, U_test):
        """Forward pass on test data (teacher-forced, not autoregressive)."""
        alaph = self.galaph
        total_neurons = sum(self.X_dim)
        X_test = np.zeros((total_neurons, U_test.shape[1]))
        for i in range(U_test.shape[1]):
            X_test[:, i:i + 1] = self.UspanX(U_test[:, i:i + 1], alaph)
        Y_test = self.Wout @ X_test
        return Y_test, X_test

    def Validate_test_data_autoregressive(self, first_input, n_steps):
        """Autoregressive forward pass: feed own predictions back as input.

        Args:
            first_input: Initial input, shape (input_dim, 1)
            n_steps: Number of timesteps to predict

        Returns:
            Y_pred: Predicted outputs, shape (output_dim, n_steps)
        """
        alaph = self.galaph
        current_input = first_input.copy()
        Y_pred = np.zeros((self.Y_dim, n_steps))
        for i in range(n_steps):
            x = self.UspanX(current_input, alaph)
            y = self.Wout @ x
            Y_pred[:, i:i + 1] = y
            current_input = y[:self.U_dim, :]
        return Y_pred

    # ------------------------------------------------------------------ #
    #  Neuron merging & pruning (evolution phase)                          #
    # ------------------------------------------------------------------ #

    def CCN_Merge_Top(self, lev, indi, indj, Q2=0):
        """Merge neurons indi and indj in layer lev (physical delete)."""
        wii = self.GroupW[lev][indi][indi]
        wij = self.GroupW[lev][indi][indj]
        wji = self.GroupW[lev][indj][indi]
        wjj = self.GroupW[lev][indj][indj]
        a, b = 0.5, 0.5
        p = 0.5 * (wii + wij + wji + wjj)

        if Q2:
            det = (wjj - wii) ** 2 + 4 * wji * wij
            if wij == 0:
                if wii != wjj:
                    rateab = wji / (wjj - wii)
                    b = 1 / (rateab + 1)
                    a = 1 - b
                    p = rateab * wij + wjj
            else:
                if det >= 0:
                    ra = (np.sqrt(det) - wjj + wii) / (2 * wij)
                    rb = (-np.sqrt(det) - wjj + wii) / (2 * wij)
                    rateab = min(ra, rb)
                    b = 1 / (rateab + 1)
                    a = 1 - b
                    p = rateab * wij + wjj

        # Input terminal
        if lev == 0:
            for i in range(self.U_dim):
                self.GroupWin[lev][indi][i] = (
                    a * self.GroupWin[lev][indi][i] + b * self.GroupWin[lev][indj][i])
        else:
            for i in range(self.X_dim[lev - 1]):
                self.GroupC[lev - 1][indi][i] = (
                    a * self.GroupC[lev - 1][indi][i] + b * self.GroupC[lev - 1][indj][i])

        # Output terminal
        if self.Stack - 1 != lev:
            for i in range(self.X_dim[lev + 1]):
                self.GroupC[lev][i][indi] = (
                    self.GroupC[lev][i][indi] + self.GroupC[lev][i][indj])

        # Recurrent matrix
        for i in range(self.X_dim[lev]):
            self.GroupW[lev][indi][i] = (
                a * self.GroupW[lev][indi][i] + b * self.GroupW[lev][indj][i])
        for i in range(self.X_dim[lev]):
            self.GroupW[lev][i][indi] = (
                self.GroupW[lev][i][indi] + self.GroupW[lev][i][indj])

        self.GroupW[lev][indi][indi] = p
        # Physical delete
        self._physical_delete_node(lev, indj)

    def CCN_Cut(self, lev, ind):
        """Prune (physically remove) neuron ind from layer lev."""
        self._physical_delete_node(lev, ind)

    def _physical_delete_node(self, lev, ind):
        """Remove a neuron by swapping with last and shrinking matrices."""
        # State vector
        self.GroupX[lev] = self._delete_rc(self.GroupX[lev], ind, -1)
        # Input weights
        if lev == 0:
            self.GroupWin[0] = self._delete_rc(self.GroupWin[0], ind, -1)
        else:
            self.GroupC[lev - 1] = self._delete_rc(self.GroupC[lev - 1], ind, -1)
        # Recurrent weights
        self.GroupW[lev] = self._delete_rc(self.GroupW[lev], ind, ind)
        # Output connection to next layer
        if lev < self.Stack - 1:
            self.GroupC[lev] = self._delete_rc(self.GroupC[lev], -1, ind)
        self.X_dim[lev] -= 1

    @staticmethod
    def _delete_rc(arr, r, c):
        """Delete row r and/or column c by swapping with last and truncating."""
        if c >= 0:
            arr[:, c] = arr[:, -1]
        if r >= 0:
            arr[r, :] = arr[-1, :]
        if r >= 0 and c >= 0:
            return arr[:-1, :-1]
        elif r >= 0:
            return arr[:-1, :]
        elif c >= 0:
            return arr[:, :-1]
        return arr

    @staticmethod
    def _sparelize(Wi, x, y, spare_rate):
        """Create a binary sparsity mask."""
        z = x * y
        zr = z - int(z * spare_rate)
        flat = Wi.reshape(z)
        idx = np.argsort(flat)
        flat[idx[:zr]] = 0
        flat[idx[zr:]] = 1
        return flat.reshape(x, y)


def run_ge_desn(U_init, U_train, Y_train, U_val, Y_val, U_test, Y_test,
                pram, oram, autoregressive=False):
    """Run a single GE-DESN grow-evolve trial and return the result.

    This implements the CCN_evaluateMergeHE logic from the original code:
    for each layer, start oversized, iteratively find the most similar neuron
    pair, merge or prune it, until the target layer size is reached.

    Args:
        U_init: shape (input_dim, init_len)
        U_train: shape (input_dim, train_len)
        Y_train: shape (output_dim, train_len)
        U_val: shape (input_dim, val_len)
        Y_val: shape (output_dim, val_len)
        U_test: shape (input_dim, test_len)
        Y_test: shape (output_dim, test_len)
        pram: structural parameters dict
        oram: weight/regularization parameters dict
        autoregressive: If True, feed predictions back as input during testing.
            If False (default), use teacher-forced evaluation.

    Returns:
        Dict containing:
          'Y_pred': predicted output, shape (output_dim, test_len)
          'nrmse_per_layer': list of NRMSE after each layer's evolution (on val)
          'max_similarity_per_layer': list of final MS after each layer
          'esn': the trained EchoStateNetwork instance
    """
    target_sizes = pram['neurons_per_layer']  # list of target sizes per layer
    neurons_add = pram['neurons_add']         # extra neurons to start with
    max_layers = pram['max_layers']
    similarity_method = oram.get('similarity_method', 0)
    Q2 = oram.get('Q2', 0)

    esn = EchoStateNetwork(U_init, U_train, Y_train, U_val, Y_val,
                           U_test, Y_test, pram, oram)

    nrmse_per_layer = []
    ms_per_layer = []

    # --- First layer ---
    esn.Inilize_First_reservoir(neurons_add + target_sizes[0])
    _evolve_layer(esn, layer_idx=0, target_size=target_sizes[0],
                  neurons_to_remove=neurons_add,
                  similarity_method=similarity_method, Q2=Q2)

    nrmse_val, ms_val = _evaluate_after_evolution(esn, autoregressive)
    nrmse_per_layer.append(nrmse_val)
    ms_per_layer.append(ms_val)

    # --- Subsequent layers ---
    for i in range(1, max_layers):
        X_num_prior = sum(esn.X_dim)
        esn.Inilize_Stack_a_reservoir(neurons_add + target_sizes[i])
        _evolve_layer(esn, layer_idx=i, target_size=target_sizes[i],
                      neurons_to_remove=neurons_add,
                      similarity_method=similarity_method, Q2=Q2,
                      protect_below=X_num_prior)

        nrmse_val, ms_val = _evaluate_after_evolution(esn, autoregressive)
        nrmse_per_layer.append(nrmse_val)
        ms_per_layer.append(ms_val)

    # Final prediction
    esn.Init_reservior(esn.U_init)
    esn.Train_reservoir(esn.U_train, esn.Y_train)
    esn.Reinit_reservoir()
    if autoregressive:
        # Drive reservoir through training data to reach correct state,
        # then predict autoregressively on the test set
        for i in range(esn.U_train.shape[1]):
            esn.UspanX(esn.U_train[:, i:i + 1], esn.galaph)
        # Snapshot pre-test reservoir state so eval can restore it without
        # re-running training (whose Wout re-solve is BLAS-nondeterministic
        # across processes and gets amplified by the AR rollout).
        esn.PreTestX = [esn.GroupX[i].copy() for i in range(esn.Stack)]
        Y_pred = esn.Validate_test_data_autoregressive(
            esn.U_test[:, 0:1], esn.U_test.shape[1])
    else:
        Y_pred, _ = esn.Validate_test_data_constant(esn.U_test)

    return {
        'Y_pred': Y_pred,
        'nrmse_per_layer': nrmse_per_layer,
        'max_similarity_per_layer': ms_per_layer,
        'esn': esn,
    }


def _evolve_layer(esn, layer_idx, target_size, neurons_to_remove,
                  similarity_method, Q2, protect_below=0):
    """Run the evolution phase for one layer: iteratively merge/prune neurons."""
    for _ in range(neurons_to_remove):
        esn.Init_reservior(esn.U_init)
        X_train = esn.Train_reservoir(esn.U_train, esn.Y_train)
        X_num, T_num = X_train.shape

        SM = Entropy.SMEstimater(X_train, X_num, T_num, similarity_method)

        # Protect neurons in earlier layers from being selected
        if protect_below > 0:
            SM[:protect_below, :protect_below] = 0

        max_sm = np.amax(SM)
        winner = np.argwhere(SM == max_sm)
        mergei, mergej = winner[0]

        esn.Reinit_reservoir()

        if protect_below > 0:
            # Higher-layer logic: cut or merge depending on where the pair is
            if mergei < protect_below:
                esn.CCN_Cut(esn.Stack - 1, mergej - protect_below)
            elif mergej < protect_below:
                esn.CCN_Cut(esn.Stack - 1, mergei - protect_below)
            else:
                esn.CCN_Merge_Top(esn.Stack - 1,
                                  mergei - protect_below,
                                  mergej - protect_below, Q2)
        else:
            esn.CCN_Merge_Top(0, mergei, mergej, Q2)


def _evaluate_after_evolution(esn, autoregressive=False):
    """Train and evaluate the ESN on validation data after an evolution step. Returns (nrmse, last_max_similarity)."""
    esn.Init_reservior(esn.U_init)
    X_train = esn.Train_reservoir(esn.U_train, esn.Y_train)
    esn.Reinit_reservoir()
    if autoregressive:
        for i in range(esn.U_train.shape[1]):
            esn.UspanX(esn.U_train[:, i:i + 1], esn.galaph)
        Yout = esn.Validate_test_data_autoregressive(
            esn.U_val[:, 0:1], esn.U_val.shape[1])
    else:
        Yout, _ = esn.Validate_test_data_constant(esn.U_val)

    err = (Yout - esn.Y_val) ** 2
    Ynorm = np.mean(esn.Y_val) * np.ones(Yout.shape)
    nerr = (Yout - Ynorm) ** 2

    denom = np.sum(nerr)
    if denom == 0:
        nrmse = 0.0
    else:
        nrmse = np.sqrt(np.sum(err) / denom)

    # Compute final max similarity
    X_num, T_num = X_train.shape
    SM = Entropy.SMEstimater(X_train, X_num, T_num, 0)
    ms = np.amax(SM)

    return float(nrmse), float(ms)
