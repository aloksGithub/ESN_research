"""
Generate an extended DDE dataset that matches the original Neutral_normed_2801.npy
and continues it with additional data points for use as a held-out test set.

Recipe (discovered via reconstruction):
- RNG: numpy.random.RandomState(seed=23)
- Initial conditions: rng.normal(0, 1, 6)
- Integration: dt=0.1, skip first 1 step
- Normalization: global z-score (mean/std across all elements of the 2801x6 window)

For the extension, we apply the SAME mean/std from the original 2801 window
so that existing train/val data is unchanged.
"""

import numpy as np
from jitcdde import jitcdde, y, past_y, past_dy, current_y, t, anchors
from symengine import Symbol, exp, sqrt, tanh


def build_dde():
    sech = lambda x: 2 / (exp(x) + exp(-x))
    eps = 1e-5
    abs_smooth = lambda x: sqrt(x**2 + eps**2)

    epsilon = [0.03966, 0.03184, 0.02847]
    nu = [1, 2.033, 3.066]
    mu = [0.16115668456085775, 0.14093420256851111, 0.11465065353644151]
    ybar_0 = 0
    tau = 1.7735
    zeta = [0.017940997406325931, 0.015689701773967984, 0.012763648066925721]

    anchors_past = Symbol("anchors_past")
    difference = Symbol("difference")
    factor_mu = Symbol("factor_mu")
    factor_zeta = Symbol("factor_zeta")

    ydot = [y(i) for i in range(3, 6)]
    y_tot = sum(current_y(i) for i in range(3))
    ydot_tot = sum(current_y(i) for i in range(3, 6))
    y_past = sum(past_y(t - tau, i, anchors_past) for i in range(3))
    ydot_past = sum(past_y(t - tau, i, anchors_past) for i in range(3, 6))
    yddot_past = sum(past_dy(t - tau, i, anchors_past) for i in range(3, 6))

    helpers = {
        (anchors_past, anchors(t - tau)),
        (difference, ybar_0 - y_past),
        (factor_mu, sech(difference) ** 2 * (yddot_past + 2 * ydot_past**2 * tanh(difference))),
        (factor_zeta, 2 * abs_smooth(y_tot) * ydot_tot),
    }

    f = {y(i): ydot[i] for i in range(3)}
    f.update({
        ydot[i]: mu[i] * factor_mu - zeta[i] * factor_zeta - epsilon[i] * nu[i] * ydot[i] - nu[i] ** 2 * y(i)
        for i in range(3)
    })

    return jitcdde(f, helpers=helpers, verbose=False)


def generate(n_steps, dt=0.1):
    rng = np.random.RandomState(23)
    DDE = build_dde()
    DDE.constant_past(rng.normal(0, 1, 6))
    DDE.adjust_diff()

    results = []
    for time_val in DDE.t + np.arange(dt, dt * (n_steps + 1), dt):
        results.append(DDE.integrate(time_val))
    return np.array(results)


if __name__ == "__main__":
    # Load original for verification and to extract normalization stats
    original = np.load("./data/Neutral_normed_2801.npy")
    print(f"Original shape: {original.shape}")

    # Generate enough data: 2801 original + 1 skipped + buffer for test set
    # Generate 4500 steps to have plenty of room (skip 1 -> 4499 usable)
    n_total_steps = 4500
    raw = generate(n_total_steps, dt=0.1)
    print(f"Raw generated shape: {raw.shape}")

    # Skip first step (index 0), take from index 1 onward
    raw_usable = raw[1:]
    print(f"Usable shape (after skip=1): {raw_usable.shape}")

    # The first 2801 points correspond to the original file
    raw_original_window = raw_usable[:2801]

    # Compute normalization stats from the original 2801 window (raw, pre-normalization)
    norm_mean = raw_original_window.mean()
    norm_std = raw_original_window.std()
    print(f"Normalization stats from 2801 window: mean={norm_mean:.10f}, std={norm_std:.10f}")

    # Normalize the original window and verify it matches
    normed_original = (raw_original_window - norm_mean) / norm_std
    max_diff = np.max(np.abs(normed_original - original))
    mean_diff = np.mean(np.abs(normed_original - original))
    print(f"\nVerification against original file:")
    print(f"  Max abs diff:  {max_diff:.10f}")
    print(f"  Mean abs diff: {mean_diff:.10f}")
    print(f"  Global mean of normed: {normed_original.mean():.10f} (expected ~0)")
    print(f"  Global std of normed:  {normed_original.std():.10f} (expected ~1)")

    if max_diff < 0.001:
        print("  *** MATCH CONFIRMED ***")
    else:
        print("  WARNING: Larger than expected differences. Check solver/platform.")

    # Normalize the FULL extended series using the SAME mean/std from the 2801 window
    normed_extended = (raw_usable - norm_mean) / norm_std
    print(f"\nExtended dataset shape: {normed_extended.shape}")
    print(f"  First 2801 points: training/validation data (unchanged)")
    print(f"  Points 2801+: available for test set")

    # Save
    output_path = "./data/Neutral_normed_extended.npy"
    np.save(output_path, normed_extended)
    print(f"\nSaved to {output_path}")
    print(f"  Total points: {len(normed_extended)}")
    print(f"  Available for test: {len(normed_extended) - 2801}")

    # Quick sanity check on the extension
    print(f"\nExtension sanity check:")
    print(f"  Last row of original window:  {normed_extended[2800]}")
    print(f"  First row of extension:       {normed_extended[2801]}")
    print(f"  Reference last row:           {original[2800]}")
