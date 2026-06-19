"""
Diagnostic: Compare v0.3 vs v0.4 reservoirpy on the gesture recognition task.

Creates a v0.3 model, extracts its exact weight matrices, copies them into a
v0.4 model, then compares:
  1. Reservoir states (should be identical if step function matches)
  2. Ridge regression solutions (Wout, bias)
  3. Final predictions and F1 scores

This isolates whether the F1 drop is caused by:
  (a) Different random weight initialization (mat_gen algorithm change), or
  (b) A difference in the Ridge regression / Model implementation.
"""

import os
import sys
import numpy as np
import warnings
import sklearn.metrics

warnings.filterwarnings("ignore")

# --- Setup paths ---
script_dir = os.path.abspath(os.path.dirname(__file__))
root_dir = os.path.dirname(script_dir)
sys.path.insert(0, root_dir)
sys.path.insert(0, os.path.join(root_dir, 'gesture_recognition'))

data_dir = os.path.join(root_dir, 'gesture_recognition', 'dataSets')

from gesture_recognition.DataSet import UniHHIMUGestures
from gesture_recognition.Utils import getData
from gesture_recognition import Evaluation
from torch.utils.data import DataLoader

# ============================================================
# Helper functions (same as esnas_gestures.py)
# ============================================================

files = ['s', 'j', 'na', 'l', 'ni']

def createData(inputFiles, testFiles, dataDir='dataSets/'):
    trainset = UniHHIMUGestures(dataDir=dataDir, train=True,
                                inputFiles=inputFiles, testFiles=testFiles,
                                useNormalized=2, learnTreshold=False,
                                shuffle=True, nFolds=len(inputFiles))
    testset = UniHHIMUGestures(dataDir=dataDir, train=False,
                               inputFiles=inputFiles, testFiles=testFiles,
                               useNormalized=2, learnTreshold=False,
                               shuffle=True, nFolds=len(inputFiles))
    trainloader = DataLoader(trainset, batch_size=1, shuffle=True, num_workers=1)
    testloader = DataLoader(testset, batch_size=1, shuffle=True, num_workers=1)
    return trainset, testset, trainloader, testloader


def evaluate_esn(esn, testFiles, label=""):
    """Run testESN-style evaluation and return F1."""
    _, _, trainloader, _ = createData(inputFiles=testFiles, testFiles=testFiles, dataDir=data_dir)
    f1_scores = []
    for test_inputs, test_targets in trainloader:
        inputs = test_inputs[0].numpy()
        targets = test_targets[0].numpy()
        prediction = esn.run(inputs)
        # Convert JAX arrays to numpy if needed
        prediction = np.asarray(prediction)
        threshold = np.ones((prediction.shape[0], 1)) * 0.4
        t_maxApp_prediction = Evaluation.calcMaxActivityPrediction(prediction, targets, threshold, 10)
        pred_MaxApp, targ_MaxApp = Evaluation.calcInputSegmentSeries(t_maxApp_prediction, targets, 0.5)
        f1_scores.append(np.mean(sklearn.metrics.f1_score(targ_MaxApp, pred_MaxApp, average=None)))
    avg_f1 = np.mean(f1_scores)
    print(f"  [{label}] F1: {avg_f1:.4f}")
    return avg_f1


# ============================================================
# Load v0.3 and v0.4 libraries side-by-side
# ============================================================

def load_v03():
    """Import v0.3 Reservoir and Ridge."""
    v03_path = os.path.join(root_dir, 'reservoirpy_versions', 'reservoirpy-0.3')
    # Temporarily insert at front
    sys.path.insert(0, v03_path)
    import importlib
    # Force fresh import
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('reservoirpy'):
            del sys.modules[mod_name]
    import reservoirpy as rpy03
    rpy03.set_seed(0)
    from reservoirpy.nodes import Reservoir, Ridge
    sys.path.remove(v03_path)
    return Reservoir, Ridge, rpy03


def load_v04():
    """Import v0.4 Reservoir and Ridge (numpy backend)."""
    v04_path = os.path.join(root_dir, 'reservoirpy_versions', 'reservoirpy-0.4')
    sys.path.insert(0, v04_path)
    import importlib
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('reservoirpy'):
            del sys.modules[mod_name]
    import reservoirpy as rpy04
    from reservoirpy.nodes import Reservoir, Ridge
    sys.path.remove(v04_path)
    return Reservoir, Ridge, rpy04


def load_v04_jax(enable_x64=False):
    """Import v0.4 Reservoir and Ridge (JAX backend)."""
    import jax
    jax.config.update("jax_enable_x64", enable_x64)

    v04_path = os.path.join(root_dir, 'reservoirpy_versions', 'reservoirpy-0.4')
    sys.path.insert(0, v04_path)
    for mod_name in list(sys.modules.keys()):
        if mod_name.startswith('reservoirpy'):
            del sys.modules[mod_name]
    import reservoirpy as rpy04
    from reservoirpy.jax.nodes import Reservoir, Ridge
    sys.path.remove(v04_path)
    return Reservoir, Ridge, rpy04


# ============================================================
# Main diagnostic
# ============================================================

def main():
    import random
    idx = 2  # fold for 'na' test subject
    inputFiles = files[:idx] + files[idx+1:]
    testFiles = files[idx:idx+1]
    print(f"Train: {inputFiles}, Test: {testFiles}")

    params = {'units': 577, 'lr': 0.7261, 'sr': 1.172, 'ridge': 7.435e-05}

    # ---- Seed and prepare data ----
    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader_v03, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX_v03, trainY_v03 = getData(trainloader_v03)

    # ============================================================
    # Test 1: v0.3 baseline (its own random weights)
    # ============================================================
    print("\n=== Test 1: v0.3 with its own weights ===")
    Reservoir03, Ridge03, rpy03 = load_v03()
    rpy03.set_seed(0)

    random.seed(42 + idx)
    np.random.seed(42 + idx)

    res03 = Reservoir03(units=params['units'], lr=params['lr'], sr=params['sr'])
    ridge03 = Ridge03(output_dim=10, ridge=params['ridge'])
    esn03 = res03 >> ridge03

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn03.fit(trainX, trainY, warmup=100)

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_v03 = evaluate_esn(esn03, testFiles, label="v0.3 own weights")

    # Extract v0.3 weight matrices (may be sparse, so convert to dense)
    from scipy.sparse import issparse

    def to_dense(m):
        """Convert any matrix type to a dense numpy ndarray."""
        if issparse(m):
            return np.asarray(m.todense())
        if hasattr(m, 'A'):  # numpy matrix
            return np.asarray(m.A)
        return np.asarray(m)

    W_v03 = to_dense(res03.W)
    Win_v03 = to_dense(res03.Win)
    bias_v03_raw = res03.bias
    Wout_v03 = to_dense(ridge03.Wout)
    bias_ridge_v03 = to_dense(ridge03.bias)

    print(f"  v0.3 W shape: {W_v03.shape} (orig type: {type(res03.W).__name__})")
    print(f"  v0.3 Win shape: {Win_v03.shape} (orig type: {type(res03.Win).__name__})")
    print(f"  v0.3 bias type: {type(bias_v03_raw).__name__}, repr: {repr(bias_v03_raw)[:100]}")
    print(f"  v0.3 Wout shape: {Wout_v03.shape}, ridge bias shape: {bias_ridge_v03.shape}")

    # Handle bias: v0.3 uses (units,1), v0.4 uses (units,)
    # If bias is still a callable or scalar, create a proper vector
    bias_v03 = to_dense(bias_v03_raw)
    if bias_v03.ndim == 0:
        # Scalar bias — v0.3 may have stored it oddly
        print(f"  WARNING: bias is scalar ({bias_v03}), expanding to (units,) zeros")
        bias_v03_flat = np.zeros(params['units'])
    else:
        bias_v03_flat = bias_v03.flatten()
    print(f"  bias_v03_flat shape: {bias_v03_flat.shape}")

    # ============================================================
    # Test 2: v0.4 with its own weights (same hyperparams)
    # ============================================================
    print("\n=== Test 2: v0.4 with its own weights ===")
    Reservoir04, Ridge04, rpy04 = load_v04()

    random.seed(42 + idx)
    np.random.seed(42 + idx)

    res04 = Reservoir04(units=params['units'], lr=params['lr'], sr=params['sr'])
    ridge04 = Ridge04(output_dim=10, ridge=params['ridge'])
    esn04 = res04 >> ridge04

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn04.fit(trainX, trainY, warmup=100)

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_v04_own = evaluate_esn(esn04, testFiles, label="v0.4 own weights")

    Wout_v04 = np.array(res04.Wout) if hasattr(res04, 'Wout') else None
    print(f"  v0.4 Wout shape: {ridge04.Wout.shape}, ridge bias shape: {ridge04.bias.shape}")

    # ============================================================
    # Test 3: v0.4 with v0.3's exact weight matrices
    # ============================================================
    print("\n=== Test 3: v0.4 with v0.3's weight matrices ===")
    Reservoir04b, Ridge04b, rpy04b = load_v04()

    # Create v0.4 reservoir with v0.3's exact matrices
    res04b = Reservoir04b(units=params['units'], lr=params['lr'], sr=params['sr'],
                          W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    ridge04b = Ridge04b(output_dim=10, ridge=params['ridge'])
    esn04b = res04b >> ridge04b

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn04b.fit(trainX, trainY, warmup=100)

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_v04_v03weights = evaluate_esn(esn04b, testFiles, label="v0.4 with v0.3 weights")

    # ============================================================
    # Test 4: Compare reservoir states directly
    # ============================================================
    print("\n=== Test 4: Compare reservoir states step-by-step ===")
    # Re-create both models with same weights to compare states
    Reservoir03c, Ridge03c, rpy03c = load_v03()
    rpy03c.set_seed(0)
    res03c = Reservoir03c(units=params['units'], lr=params['lr'], sr=params['sr'],
                          W=W_v03, Win=Win_v03, bias=bias_v03)
    states_v03 = res03c.run(trainX[:200])  # first 200 steps for speed

    Reservoir04c, Ridge04c, rpy04c = load_v04()
    res04c = Reservoir04c(units=params['units'], lr=params['lr'], sr=params['sr'],
                          W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    states_v04 = res04c.run(trainX[:200])

    states_v03 = np.array(states_v03)
    states_v04 = np.array(states_v04)

    max_diff = np.max(np.abs(states_v03 - states_v04))
    mean_diff = np.mean(np.abs(states_v03 - states_v04))
    print(f"  Reservoir state max diff:  {max_diff:.2e}")
    print(f"  Reservoir state mean diff: {mean_diff:.2e}")

    # ============================================================
    # Test 5: Compare Ridge solutions with identical reservoir states
    # ============================================================
    print("\n=== Test 5: Compare Ridge with identical states ===")
    # Train both ridges on the SAME reservoir states (from v0.3 reservoir)
    Reservoir03d, Ridge03d, rpy03d = load_v03()
    rpy03d.set_seed(0)
    res03d = Reservoir03d(units=params['units'], lr=params['lr'], sr=params['sr'],
                          W=W_v03, Win=Win_v03, bias=bias_v03)
    ridge03d = Ridge03d(output_dim=10, ridge=params['ridge'])
    esn03d = res03d >> ridge03d

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn03d.fit(trainX, trainY, warmup=100)

    Reservoir04d, Ridge04d, rpy04d = load_v04()
    res04d = Reservoir04d(units=params['units'], lr=params['lr'], sr=params['sr'],
                          W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    ridge04d = Ridge04d(output_dim=10, ridge=params['ridge'])
    esn04d = res04d >> ridge04d

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn04d.fit(trainX, trainY, warmup=100)

    Wout_03d = np.asarray(ridge03d.Wout)
    Wout_04d = np.asarray(ridge04d.Wout)
    bias_03d = np.asarray(ridge03d.bias).flatten()
    bias_04d = np.asarray(ridge04d.bias).flatten()

    print(f"  v0.3 Wout shape: {Wout_03d.shape}, bias shape: {bias_03d.shape}")
    print(f"  v0.4 Wout shape: {Wout_04d.shape}, bias shape: {bias_04d.shape}")
    print(f"  Wout max diff:  {np.max(np.abs(Wout_03d - Wout_04d)):.2e}")
    print(f"  Wout mean diff: {np.mean(np.abs(Wout_03d - Wout_04d)):.2e}")
    print(f"  Bias max diff:  {np.max(np.abs(bias_03d - bias_04d)):.2e}")
    print(f"  Bias mean diff: {np.mean(np.abs(bias_03d - bias_04d)):.2e}")

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_v03_shared = evaluate_esn(esn03d, testFiles, label="v0.3 ridge (shared weights)")
    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_v04_shared = evaluate_esn(esn04d, testFiles, label="v0.4 ridge (shared weights)")

    # ============================================================
    # Test 6: v0.4 JAX (float32) with v0.3's weight matrices
    # ============================================================
    print("\n=== Test 6: v0.4 JAX (float32, no x64) with v0.3 weights ===")
    import jax
    import jax.numpy as jnp

    Reservoir04_jax32, Ridge04_jax32, _ = load_v04_jax(enable_x64=False)

    res04_jax32 = Reservoir04_jax32(units=params['units'], lr=params['lr'], sr=params['sr'],
                                     W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    ridge04_jax32 = Ridge04_jax32(output_dim=10, ridge=params['ridge'])
    esn04_jax32 = res04_jax32 >> ridge04_jax32

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn04_jax32.fit(trainX, trainY, warmup=100)

    # Compare reservoir states (JAX float32 vs numpy float64)
    Reservoir04_jax32b, _, _ = load_v04_jax(enable_x64=False)
    res04_jax32b = Reservoir04_jax32b(units=params['units'], lr=params['lr'], sr=params['sr'],
                                      W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    states_jax32 = np.asarray(res04_jax32b.run(trainX[:200]))
    jax32_state_max_diff = np.max(np.abs(states_v04 - states_jax32))
    jax32_state_mean_diff = np.mean(np.abs(states_v04 - states_jax32))
    print(f"  JAX float32 vs numpy float64 state max diff:  {jax32_state_max_diff:.2e}")
    print(f"  JAX float32 vs numpy float64 state mean diff: {jax32_state_mean_diff:.2e}")

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_jax32 = evaluate_esn(esn04_jax32, testFiles, label="v0.4 JAX float32 + v0.3 weights")

    # ============================================================
    # Test 7: v0.4 JAX (float64) with v0.3's weight matrices
    # ============================================================
    print("\n=== Test 7: v0.4 JAX (float64, x64 enabled) with v0.3 weights ===")
    Reservoir04_jax64, Ridge04_jax64, _ = load_v04_jax(enable_x64=True)

    res04_jax64 = Reservoir04_jax64(units=params['units'], lr=params['lr'], sr=params['sr'],
                                     W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    ridge04_jax64 = Ridge04_jax64(output_dim=10, ridge=params['ridge'])
    esn04_jax64 = res04_jax64 >> ridge04_jax64

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    _, _, trainloader, _ = createData(inputFiles=inputFiles, testFiles=testFiles, dataDir=data_dir)
    trainX, trainY = getData(trainloader)
    esn04_jax64.fit(trainX, trainY, warmup=100)

    # Compare reservoir states (JAX float64 vs numpy float64)
    Reservoir04_jax64b, _, _ = load_v04_jax(enable_x64=True)
    res04_jax64b = Reservoir04_jax64b(units=params['units'], lr=params['lr'], sr=params['sr'],
                                      W=W_v03, Win=Win_v03, bias=bias_v03_flat)
    states_jax64 = np.asarray(res04_jax64b.run(trainX[:200]))
    jax64_state_max_diff = np.max(np.abs(states_v04 - states_jax64))
    jax64_state_mean_diff = np.mean(np.abs(states_v04 - states_jax64))
    print(f"  JAX float64 vs numpy float64 state max diff:  {jax64_state_max_diff:.2e}")
    print(f"  JAX float64 vs numpy float64 state mean diff: {jax64_state_mean_diff:.2e}")

    random.seed(42 + idx)
    np.random.seed(42 + idx)
    f1_jax64 = evaluate_esn(esn04_jax64, testFiles, label="v0.4 JAX float64 + v0.3 weights")

    # ============================================================
    # Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  v0.3 own weights:           F1 = {f1_v03:.4f}")
    print(f"  v0.4 own weights:           F1 = {f1_v04_own:.4f}")
    print(f"  v0.4 with v0.3 weights:     F1 = {f1_v04_v03weights:.4f}")
    print(f"  v0.3 ridge (shared W):      F1 = {f1_v03_shared:.4f}")
    print(f"  v0.4 ridge (shared W):      F1 = {f1_v04_shared:.4f}")
    print(f"  v0.4 JAX float32 + v0.3 W:  F1 = {f1_jax32:.4f}")
    print(f"  v0.4 JAX float64 + v0.3 W:  F1 = {f1_jax64:.4f}")
    print()
    if max_diff < 1e-10:
        print("  v0.3 vs v0.4 numpy states: IDENTICAL")
    else:
        print(f"  v0.3 vs v0.4 numpy states: DIFFER (max diff = {max_diff:.2e})")
    print(f"  JAX float32 vs numpy state max diff: {jax32_state_max_diff:.2e}")
    print(f"  JAX float64 vs numpy state max diff: {jax64_state_max_diff:.2e}")
    print()
    if f1_v04_v03weights > 0.85:
        print("  CONCLUSION: v0.4 numpy is correct — F1 drop was due to different random weights.")
    if f1_jax64 > 0.85:
        print("  CONCLUSION: v0.4 JAX float64 is also correct.")
    if f1_jax32 < 0.85 and f1_jax64 > 0.85:
        print("  CONCLUSION: JAX float32 degrades performance — enable x64 for ESNAS!")
    elif f1_jax32 > 0.85:
        print("  CONCLUSION: JAX float32 is fine too — precision is not the issue.")


if __name__ == "__main__":
    main()
