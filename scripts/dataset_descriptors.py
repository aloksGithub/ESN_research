
import sys
import os

from statsmodels.tsa.stattools import adfuller, kpss

current_dir = os.path.abspath(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
import pickle
from src.algorithms.ESN_GA import ESN_GA
from src.datasets import getDataLaser, getDataMGS, getDataLorenz, getDataDDE, getDataSunspots, getDataWater
import numpy as np
import reservoirpy as rpy
import pandas as pd
import json
import antropy as ant
from hurst import compute_Hc
import ruptures as rpt
from scipy import stats
from statsmodels.stats.multitest import multipletests
import statsmodels.api as sm
import traceback

from src.utils import constructModel, runModel
rpy.verbosity(0)
import warnings
warnings.filterwarnings("ignore")

def compute_dataset_descriptors(max_lag: int = 100, n_bins: int = 16) -> pd.DataFrame:
    """Compute dataset-level descriptors for all datasets.

    Returns a DataFrame with one row per dataset containing:
    - permutation_entropy, sample_entropy, spectral_entropy, lz_complexity, hurst
    - ami_first_min (lag of first local minimum in auto mutual information)
    - seasonality_strength, dominant_period_samples
    - adf_pvalue, kpss_pvalue (if statsmodels available)
    - change_point_count (if ruptures available)
    - snr_ma (variance ratio using moving-average residual)
    - series_length, num_variables
    """

    dataset_funcs = [
        ("laser", getDataLaser),
        ("mgs", getDataMGS),
        ("lorenz", getDataLorenz),
        ("dde", getDataDDE),
        ("sunspots", getDataSunspots),
        ("water", getDataWater),
    ]

    rows = []
    for name, fn in dataset_funcs:
        try:
            trainX, trainY, valX, valY, testX, testY = fn()
        except Exception:
            # Fallback to older signature if necessary
            _, trainY, _, _, _, _ = fn()
            trainX = None
        y = trainY
        if y is None:
            continue
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        series_length = int(y.shape[0])
        num_variables = int(y.shape[1])

        # Compute per-channel metrics, then aggregate by median
        pe = []
        se = []
        spen = []
        lzc = []
        hur = []
        ami_min = []
        seas = []
        domp = []
        adf_ps = []
        kpss_ps = []
        cps = []
        snr_vals = []

        for j in range(num_variables):
            x = np.asarray(y[:, j], dtype=float)
            x = x[~np.isnan(x)]
            if x.size < 64:
                continue
            # Antropy-based descriptors
            pe.append(float(ant.perm_entropy(x, order=3, delay=1, normalize=True)))
            se.append(float(ant.sample_entropy(x)))
            spen.append(float(ant.spectral_entropy(x, sf=1.0, method='fft', normalize=True)))
            lzc.append(_lziv_from_signal(x))
            hur.append(compute_Hc(x, kind="change", simplified=True)[0])
            ami_min.append(_ami_first_min(x, max_lag=max_lag, n_bins=n_bins))
            # Compute PSD once for seasonality metrics
            sp, freq = _power_spectrum(x)
            strength, period = _seasonality_strength_and_period(sp, freq)
            seas.append(strength)
            domp.append(period)
            adf_ps.append(adfuller(x, autolag="AIC")[1])
            kpss_ps.append(kpss(x, regression="c", nlags="auto")[1])
            cps.append(_change_points_count(x))
            snr_vals.append(_snr_moving_average(x))

        rows.append({
            "dataset": name,
            "series_length": series_length,
            "num_variables": num_variables,
            "permutation_entropy": np.nanmedian(pe) if len(pe) else np.nan,
            "sample_entropy": np.nanmedian(se) if len(se) else np.nan,
            "spectral_entropy": np.nanmedian(spen) if len(spen) else np.nan,
            "lz_complexity": np.nanmedian(lzc) if len(lzc) else np.nan,
            "hurst": np.nanmedian(hur) if len(hur) else np.nan,
            "ami_first_min": np.nanmedian(ami_min) if len(ami_min) else np.nan,
            "seasonality_strength": np.nanmedian(seas) if len(seas) else np.nan,
            "dominant_period_samples": np.nanmedian(domp) if len(domp) else np.nan,
            "adf_pvalue": np.nanmedian(adf_ps) if len(adf_ps) else np.nan,
            "kpss_pvalue": np.nanmedian(kpss_ps) if len(kpss_ps) else np.nan,
            "change_point_count": np.nanmedian(cps) if len(cps) else np.nan,
            "snr_ma": np.nanmedian(snr_vals) if len(snr_vals) else np.nan,
        })

    return pd.DataFrame(rows)

def _power_spectrum(x: np.ndarray) -> tuple:
    """Return (power, freqs) for rFFT, excluding DC handling elsewhere."""
    x = np.asarray(x) - np.mean(x)
    n = x.size
    fft = np.fft.rfft(x)
    P = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0)
    return P, freqs

def _lziv_from_signal(x: np.ndarray) -> float:
    """Compute LZ complexity via Antropy by binarizing the signal."""
    x = np.asarray(x)
    if x.size < 10:
        return np.nan
    # Binarize by median
    m = np.median(x)
    s = (x > m).astype(int)
    # Convert to bitstring
    bitstr = ''.join('1' if v == 1 else '0' for v in s)
    return float(ant.lziv_complexity(bitstr, normalize=True))

def _ami_first_min(x: np.ndarray, max_lag: int = 100, n_bins: int = 16) -> float:
    """First local minimum of auto mutual information (binned)."""
    x = np.asarray(x)
    if x.size < max_lag + 10:
        max_lag = max(10, x.size // 5)
    # Uniform binning to avoid degenerate quantiles
    edges = np.linspace(np.min(x), np.max(x) + 1e-12, n_bins + 1)
    bx = np.digitize(x, edges[:-1])
    # ensure indices are within [0, n_bins-1]
    bx = np.clip(bx, 0, n_bins - 1)
    mis = []
    for lag in range(1, max_lag + 1):
        a = bx[:-lag]
        b = bx[lag:]
        mis.append(_mutual_information_discrete(a, b, n_bins))
    mis = np.array(mis)
    for i in range(1, len(mis) - 1):
        if mis[i] < mis[i - 1] and mis[i] < mis[i + 1]:
            return float(i + 1)
    # Fallback: global minimum
    return float(np.argmin(mis) + 1) if len(mis) else np.nan

def _mutual_information_discrete(a: np.ndarray, b: np.ndarray, n_bins: int) -> float:
    N = len(a)
    if N == 0:
        return np.nan
    a = np.asarray(a, dtype=int)
    b = np.asarray(b, dtype=int)
    a = np.clip(a, 0, n_bins - 1)
    b = np.clip(b, 0, n_bins - 1)
    Pa = np.bincount(a, minlength=n_bins) / N
    Pb = np.bincount(b, minlength=n_bins) / N
    Pab = np.zeros((n_bins, n_bins), dtype=float)
    for i in range(N):
        Pab[a[i], b[i]] += 1.0
    Pab /= N
    nz = Pab > 0
    # gather corresponding marginals for nonzero joint entries
    rows, cols = np.nonzero(nz)
    joint_vals = Pab[rows, cols]
    pa_vals = Pa[rows]
    pb_vals = Pb[cols]
    mi = np.sum(joint_vals * (np.log(joint_vals) - np.log(pa_vals + 1e-12) - np.log(pb_vals + 1e-12)))
    return float(mi)

def _seasonality_strength_and_period(P: np.ndarray, freq: np.ndarray) -> tuple:
    """Return (strength, dominant_period_in_samples)."""
    if len(P) < 3:
        return np.nan, np.nan
    Pn = P[1:]
    fn = freq[1:]
    total = np.sum(Pn)
    if total <= 0:
        return np.nan, np.nan
    idx = int(np.argmax(Pn))
    strength = float(Pn[idx] / total)
    f = fn[idx]
    period = float(1.0 / f) if f > 0 else np.nan
    return strength, period

def _change_points_count(x: np.ndarray) -> float:
    """Heuristic change-point count via ruptures if available."""
    n = len(x)
    model = rpt.Pelt(model="rbf").fit(x)
    pen = 3.0 * np.log(n)
    bkps = model.predict(pen=pen)
    # bkps includes end index
    return float(max(0, len(bkps) - 1))

def _snr_moving_average(x: np.ndarray) -> float:
    """Variance ratio var(x)/var(x - MA(x)) as a crude SNR proxy."""
    x = np.asarray(x)
    n = len(x)
    if n < 20:
        return np.nan
    w = max(5, int(n * 0.02))
    kernel = np.ones(w) / w
    ma = np.convolve(x, kernel, mode="same")
    resid = x - ma
    vr = np.var(x) / (np.var(resid) + 1e-12)
    return float(vr)

if __name__ == "__main__":
    # Save both datasets to CSV for downstream analysis
    out_dir = os.path.join(parent_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    # 1) Dataset-level descriptors
    ds = compute_dataset_descriptors()
    ds_path = os.path.join(out_dir, "datasets.csv")
    ds.to_csv(ds_path, index=False)
    print(f"Saved dataset descriptors to: {ds_path} (shape={ds.shape})")
    