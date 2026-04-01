"""
Generate an extended Lorenz dataset that matches Lorenz_normed_2801.npy
and continues it with additional points for a held-out test set.

Recipe (discovered via reconstruction):
- Solver: scipy.integrate.odeint (LSODA)
- dt: 0.001, subsample every 10th point (effective dt = 0.01)
- Initial conditions: (1, 1, 0)
- Parameters: sigma=10, rho=28, beta=8/3
- Skip: 0 (no transient discard)
- Normalization: global z-score on the 2801-point window, same stats applied to extension
"""

import numpy as np
from scipy.integrate import odeint


def lorenz(state, t, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z]


def global_zscore(data):
    return (data - data.mean()) / data.std()


if __name__ == "__main__":
    # Load original for verification
    original = np.load("./data/Lorenz_normed_2801.npy")
    print(f"Original shape: {original.shape}")

    y0 = [1.0, 1.0, 0.0]

    # Generate enough data: need 2801*10 = 28010 fine steps for original,
    # plus extra for test set. Generate 45000 fine steps -> 4500 subsampled.
    dt_fine = 0.001
    n_fine = 45000
    t_eval = np.arange(0, n_fine * dt_fine, dt_fine)
    sol = odeint(lorenz, y0, t_eval)

    # Subsample every 10th point
    raw = sol[::10]
    print(f"Subsampled shape: {raw.shape}")

    # The first 2801 points correspond to the original file
    raw_original_window = raw[:2801]

    # Compute normalization stats from original 2801 window
    norm_mean = raw_original_window.mean()
    norm_std = raw_original_window.std()
    print(f"Normalization stats: mean={norm_mean:.10f}, std={norm_std:.10f}")

    # Normalize and verify
    normed_original = (raw_original_window - norm_mean) / norm_std
    max_diff = np.max(np.abs(normed_original - original))
    mean_diff = np.mean(np.abs(normed_original - original))
    print(f"\nVerification against original file:")
    print(f"  Max abs diff:  {max_diff:.10f}")
    print(f"  Mean abs diff: {mean_diff:.10f}")
    print(f"  Global mean: {normed_original.mean():.10f}")
    print(f"  Global std:  {normed_original.std():.10f}")

    if max_diff < 0.001:
        print("  *** MATCH CONFIRMED ***")
    else:
        print(f"  WARNING: max diff = {max_diff:.6f}")

    print(f"\nFirst 3 rows (generated): {normed_original[:3]}")
    print(f"First 3 rows (reference): {original[:3]}")

    # Normalize full extended series with SAME mean/std
    normed_extended = (raw - norm_mean) / norm_std
    print(f"\nExtended dataset shape: {normed_extended.shape}")
    print(f"  First 2801: train/val (unchanged)")
    print(f"  Points 2801+: available for test set ({len(normed_extended) - 2801} points)")

    # Save
    output_path = "./data/Lorenz_normed_extended.npy"
    np.save(output_path, normed_extended)
    print(f"\nSaved to {output_path}")

    # Sanity check continuity
    print(f"\nContinuity check:")
    print(f"  Last original row:    {normed_extended[2800]}")
    print(f"  First extension row:  {normed_extended[2801]}")
    print(f"  Reference last row:   {original[2800]}")
