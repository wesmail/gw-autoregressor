#!/usr/bin/env python3
"""
evaluate_overlap.py — Autoregressive & teacher-forced waveform evaluation
==========================================================================

Computes **maximised noise-weighted overlap** (match/mismatch) between
predicted and ground-truth gravitational waveforms for a causal transformer
surrogate model.

Two evaluation modes
--------------------
1. **Teacher-forced** (``--mode teacher``):  Feed ground-truth input tokens
   and compare the model's one-step predictions with the target.

2. **Autoregressive rollout** (``--mode rollout``):  Given a *context window*
   of ``--context_s`` seconds, the model predicts the next ``--future_s``
   seconds token-by-token, feeding its own predictions back as input.  This
   is the true test of a conditional-continuation surrogate: the context
   covers the inspiral and the model must generate pre-merger → merger →
   ringdown autoregressively.

PSD support
-----------
* ``--psd csv`` + ``--psd_file path.csv``:  load a two-column CSV
  (``frequency``, ``psd``) and interpolate in log-log space — matching the
  ``ETBluebookSensitivityPSD`` function from the project's ``noise.py``.
* ``--psd et_analytic``:  built-in analytic ET-D fit (Hild et al. 2011).
* ``--psd flat``:  uniform PSD (overlap reduces to un-weighted match).

Why overlap instead of MSE?
    GW data analysis uses matched filtering; the detection statistic is the
    noise-weighted inner product.  Overlap reflects the detector's actual
    sensitivity curve and is the standard figure of merit in waveform
    modelling.  MSE weights all frequencies equally and is not physically
    meaningful for detection.

Usage examples
--------------
    # Autoregressive rollout with ET Bluebook PSD from CSV
    python evaluate_overlap.py \\
        --ckpt checkpoints/epoch=9.ckpt \\
        --data_dir /data/merger_windows \\
        --mode rollout \\
        --context_s 0.5 --future_s 0.25 \\
        --psd csv --psd_file et_bluebook.csv \\
        --fs 4096 --fmin 2 --fmax 1024 \\
        --batch_size 32 --worst_k 10 \\
        --out_dir results/autoregressive

    # Quick teacher-forced check
    python evaluate_overlap.py \\
        --ckpt checkpoints/epoch=9.ckpt \\
        --data_dir /data/merger_windows \\
        --mode teacher \\
        --psd et_analytic \\
        --max_events 500 --out_dir results/teacher
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Project imports — try several layouts, fail gracefully
# ---------------------------------------------------------------------------
_IMPORT_OK = False
for _attempt in [
    lambda: __import__("models.lightning_module", fromlist=["GPTLightning"]),
    lambda: __import__("lightning_module", fromlist=["GPTLightning"]),
]:
    try:
        _mod = _attempt()
        GPTLightning = _mod.GPTLightning
        _IMPORT_OK = True
        break
    except Exception:
        continue

for _attempt in [
    lambda: __import__("datasets.data_handling", fromlist=["GWDataModule"]),
    lambda: __import__("data_handling", fromlist=["GWDataModule"]),
]:
    try:
        _mod = _attempt()
        GWDataModule = _mod.GWDataModule
        _worker_init = _mod._worker_init
        break
    except Exception:
        continue

if not _IMPORT_OK:
    print(
        "ERROR: Cannot import GPTLightning / GWDataModule.\n"
        "Ensure the repo root is on PYTHONPATH or run from the repo directory."
    )
    sys.exit(1)


###############################################################################
# §1  PSD LOADING
###############################################################################

def load_psd_from_csv(
    freqs: np.ndarray,
    csv_path: str,
    low_freq_cutoff: float = 1.0,
) -> np.ndarray:
    """
    Load a PSD from a CSV file with columns ``frequency`` and ``psd``,
    interpolate in log-log space (matching ``ETBluebookSensitivityPSD``
    from ``noise.py``), and evaluate at the requested frequencies.

    Parameters
    ----------
    freqs : 1-D array of target frequencies [Hz]
    csv_path : path to CSV (must have 'frequency' and 'psd' columns)
    low_freq_cutoff : frequencies below this are zeroed

    Returns
    -------
    psd_values : 1-D array, same length as *freqs*
    """
    import pandas as pd
    from scipy.interpolate import interp1d

    df = pd.read_csv(csv_path)
    if "frequency" not in df.columns or "psd" not in df.columns:
        raise ValueError(
            f"CSV file {csv_path} must have 'frequency' and 'psd' columns. "
            f"Found: {list(df.columns)}"
        )
    df = df.sort_values("frequency")

    # Log-log interpolation (same as noise.py)
    log_interp = interp1d(
        np.log(df["frequency"].values),
        np.log(df["psd"].values),
        kind="linear",
        fill_value="extrapolate",
    )

    psd_values = np.zeros(len(freqs), dtype=np.float64)
    valid = freqs >= max(low_freq_cutoff, df["frequency"].min())
    psd_values[valid] = np.exp(log_interp(np.log(freqs[valid])))
    return psd_values


def _et_d_psd_analytic(freqs: np.ndarray) -> np.ndarray:
    """
    Analytic approximation of the Einstein Telescope ET-D sensitivity curve.
    Fit from Hild et al. (2011) — arXiv:1012.0908, Table 1 (xylophone).
    Accurate to ~10% in 1–10 000 Hz; sufficient for overlap computations
    where only the PSD *shape* matters.
    """
    f = np.asarray(freqs, dtype=np.float64)
    f = np.clip(f, 1e-30, None)
    f0 = 100.0
    x = f / f0
    S0 = 1.449e-52
    Sn = S0 * (
        2.39e-27 * x ** (-15.64)
        + 0.349 * x ** (-2.145)
        + 1.76 * x ** (-0.12)
        + 0.409 * x ** 1.10
    )
    return Sn


def load_psd(
    psd_type: str,
    psd_file: Optional[str],
    freqs: np.ndarray,
    low_freq_cutoff: float = 1.0,
) -> np.ndarray:
    """
    Return Sn(f) evaluated at *freqs*.

    psd_type : {"csv", "et_analytic", "flat"}
    """
    if psd_type == "flat":
        return np.ones_like(freqs, dtype=np.float64)

    if psd_type == "et_analytic":
        return _et_d_psd_analytic(freqs)

    if psd_type == "csv":
        if psd_file is None:
            raise ValueError("--psd_file is required when --psd csv")
        return load_psd_from_csv(freqs, psd_file, low_freq_cutoff)

    raise ValueError(f"Unknown PSD type '{psd_type}'")


###############################################################################
# §2  FREQUENCY-FEATURE HELPER  (mirrors MergerWindowDataset._freq_features)
###############################################################################

def compute_freq_features(
    tokens: torch.Tensor,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
) -> torch.Tensor:
    """
    Per-token FFT magnitude features.

    Parameters
    ----------
    tokens : [B, T, C, K]  or  [T, C, K]
    freq_keep_bins : number of low-frequency bins to keep
    freq_log1p : apply log1p to magnitudes

    Returns
    -------
    [B, T, C, F]  or  [T, C, F]
    """
    mag = torch.fft.rfft(tokens, dim=-1).abs()
    if freq_log1p:
        mag = torch.log1p(mag)
    Fkeep = min(freq_keep_bins, mag.shape[-1])
    return mag[..., :Fkeep]


###############################################################################
# §3  TOKEN RECONSTRUCTION
###############################################################################

def reconstruct_full_tokens(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Reconstruct the complete token sequence from dataset outputs.

    The dataset does::

        tokens = waveform.unfold(0, K, stride)   # [T+1, C, K]
        x = tokens[:-1]                           # [T, C, K]
        y = tokens[1:].permute(0,2,1).view(T*K, C)

    So the full sequence is  ``cat(x[0:1], y_as_tokens)``.

    Parameters
    ----------
    x : [T, C, K]  (unbatched) or [B, T, C, K]
    y : [T*K, C]   (unbatched) or [B, T*K, C]

    Returns
    -------
    tokens : [T+1, C, K]  or  [B, T+1, C, K]
    """
    batched = x.dim() == 4
    if not batched:
        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

    B, T, C, K = x.shape
    # y: [B, T*K, C] → [B, T, K, C] → [B, T, C, K]
    y_tok = y.view(B, T, K, C).permute(0, 1, 3, 2)  # [B, T, C, K]
    # full = first input token + all target tokens
    full = torch.cat([x[:, :1], y_tok], dim=1)  # [B, T+1, C, K]

    if not batched:
        full = full.squeeze(0)
    return full


###############################################################################
# §4  CORE OVERLAP FUNCTIONS
###############################################################################

def build_complex_strain(hp: np.ndarray, hc: np.ndarray) -> np.ndarray:
    """h(t) = h_+(t) − i h_×(t)"""
    return hp.astype(np.float64) - 1j * hc.astype(np.float64)


def compute_overlap_maximised(
    h_true_t: np.ndarray,
    h_pred_t: np.ndarray,
    fs: float,
    psd: np.ndarray,
    freqs: np.ndarray,
    fmin: float,
    fmax: float,
) -> float:
    """
    Overlap maximised over time shift *and* coalescence phase.

    1. FFT both complex strains.
    2. Frequency-domain cross-correlation via IFFT:
           z(τ) = IFFT[ h̃₁(f) · conj(h̃₂(f)) / Sn(f) ]
       |z(τ)| is the phase-maximised overlap at lag τ.
    3. Normalise by sqrt(σ₁² · σ₂²).
    4. Return max over τ.
    """
    N = len(h_true_t)
    df = fs / N

    h1f = np.fft.rfft(h_true_t)
    h2f = np.fft.rfft(h_pred_t)

    mask = (freqs >= fmin) & (freqs <= fmax)
    psd_safe = np.clip(psd, 1e-300, None)

    # Normalisation
    sig1_sq = 4.0 * df * np.sum(np.abs(h1f[mask]) ** 2 / psd_safe[mask]).real
    sig2_sq = 4.0 * df * np.sum(np.abs(h2f[mask]) ** 2 / psd_safe[mask]).real
    norm = np.sqrt(sig1_sq * sig2_sq)
    if norm < 1e-30:
        return 0.0

    # Cross-correlation via IFFT (maximises over time shift)
    integrand = np.zeros(len(h1f), dtype=np.complex128)
    integrand[mask] = h1f[mask] * np.conj(h2f[mask]) / psd_safe[mask]
    z_tau = np.fft.irfft(integrand, n=N)

    # |z(τ)| maximises over phase analytically
    overlap = 4.0 * df * np.max(np.abs(z_tau)) / norm
    return float(np.clip(overlap, 0.0, 1.0))


###############################################################################
# §5  AUTOREGRESSIVE ROLLOUT
###############################################################################

@torch.no_grad()
def autoregressive_rollout(
    model: GPTLightning,
    context_tokens: torch.Tensor,
    theta: torch.Tensor,
    n_future_tokens: int,
    max_context_window: Optional[int] = None,
    use_freq_features: bool = True,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
) -> torch.Tensor:
    """
    Generate ``n_future_tokens`` autoregressively from a context window.

    Parameters
    ----------
    model : trained GPTLightning (already on device, eval mode)
    context_tokens : [B, T_ctx, C, K]  — the conditioning context
    theta : [B, theta_dim]
    n_future_tokens : how many tokens to generate
    max_context_window : if set, use a sliding window of this many tokens to
        bound memory/compute during rollout (model has finite max_len).
    use_freq_features : if False, pass None for x_freq (uni-modal time-only model).
    freq_keep_bins, freq_log1p : frequency-feature parameters when use_freq_features=True.

    Returns
    -------
    predicted_tokens : [B, n_future_tokens, C, K]
    """
    device = context_tokens.device
    B, T_ctx, C, K = context_tokens.shape

    # Effective maximum window the model can handle
    model_max_len = model.hparams.get("max_len", 5000)
    if max_context_window is None:
        max_context_window = model_max_len
    max_win = min(max_context_window, model_max_len)

    # Working buffer — starts as context, grows with each prediction
    tokens = context_tokens.clone()  # [B, T_ctx, C, K]

    predicted: List[torch.Tensor] = []

    for step in range(n_future_tokens):
        # Sliding window: keep only the last max_win tokens
        if tokens.shape[1] > max_win:
            tokens = tokens[:, -max_win:]

        # Compute frequency features for current tokens (or None for uni-modal)
        if use_freq_features:
            x_freq = compute_freq_features(tokens, freq_keep_bins, freq_log1p)
        else:
            x_freq = None

        # Forward pass (causal)
        out = model(tokens, x_freq, is_causal=True, theta=theta)  # [B, T*K, C]

        T_cur = tokens.shape[1]
        # Reshape to [B, T_cur, K, C] and take the LAST token's prediction
        out_tok = out.view(B, T_cur, K, C)
        next_pred = out_tok[:, -1]            # [B, K, C]

        # Convert to input-token shape: [B, K, C] → [B, 1, C, K]
        next_token = next_pred.permute(0, 2, 1).unsqueeze(1)  # [B, 1, C, K]

        predicted.append(next_token)

        # Append to the running context
        tokens = torch.cat([tokens, next_token], dim=1)

        if (step + 1) % 50 == 0 or step == n_future_tokens - 1:
            print(f"    Rollout step {step + 1}/{n_future_tokens}")

    # Stack: list of [B, 1, C, K] → [B, T_fut, C, K]
    return torch.cat(predicted, dim=1)


def tokens_to_waveform(tokens: torch.Tensor) -> torch.Tensor:
    """
    Flatten a token tensor to a continuous waveform.

    Parameters
    ----------
    tokens : [B, T, C, K]  or  [T, C, K]

    Returns
    -------
    waveform : [B, T*K, C]  or  [T*K, C]
    """
    batched = tokens.dim() == 4
    if not batched:
        tokens = tokens.unsqueeze(0)
    B, T, C, K = tokens.shape
    # [B, T, C, K] → [B, T, K, C] → [B, T*K, C]
    wf = tokens.permute(0, 1, 3, 2).contiguous().view(B, T * K, C)
    if not batched:
        wf = wf.squeeze(0)
    return wf


@torch.no_grad()
def apply_stitcher_to_waveform(
    model: GPTLightning,
    waveform: torch.Tensor,
) -> torch.Tensor:
    """
    Optionally apply the model's causal stitcher to a continuous waveform
    to reduce token-boundary artifacts.

    Parameters
    ----------
    model : GPTLightning (must have model.gpt.use_stitcher == True)
    waveform : [B, L, C]

    Returns
    -------
    smoothed : [B, L, C]
    """
    if not getattr(model.gpt, "use_stitcher", False):
        return waveform
    # Stitcher expects [B, C, L]
    device = next(model.parameters()).device
    wf = waveform.to(device).transpose(1, 2)   # [B, C, L]
    wf = model.gpt.stitcher(wf)
    return wf.transpose(1, 2).cpu()             # [B, L, C]


###############################################################################
# §6  TEACHER-FORCED INFERENCE
###############################################################################

@torch.no_grad()
def run_teacher_forced_inference(
    model: GPTLightning,
    dataloader,
    device: torch.device,
    max_events: Optional[int] = None,
) -> dict:
    """
    Teacher-forced: feed ground-truth x, compare model output to y.
    """
    model.eval()
    hp_true_all, hc_true_all = [], []
    hp_pred_all, hc_pred_all = [], []
    theta_all = []
    count = 0

    for batch in dataloader:
        x = batch["x"].to(device)
        x_freq = batch.get("x_freq")
        if x_freq is not None:
            x_freq = x_freq.to(device)
        theta = batch["theta"].to(device)
        y = batch["y"]  # [B, L, C]

        y_hat = model(x, x_freq, is_causal=True, theta=theta)
        y_hat = y_hat.float().cpu().numpy()
        y_np = y.float().numpy()
        theta_np = theta.float().cpu().numpy()

        B = y_np.shape[0]
        for i in range(B):
            if max_events is not None and count >= max_events:
                break
            hp_true_all.append(y_np[i, :, 0])
            hc_true_all.append(y_np[i, :, 1])
            hp_pred_all.append(y_hat[i, :, 0])
            hc_pred_all.append(y_hat[i, :, 1])
            theta_all.append(theta_np[i])
            count += 1
        if max_events is not None and count >= max_events:
            break

    return {
        "hp_true": hp_true_all, "hc_true": hc_true_all,
        "hp_pred": hp_pred_all, "hc_pred": hc_pred_all,
        "theta": np.array(theta_all),
    }


###############################################################################
# §7  AUTOREGRESSIVE INFERENCE (FULL PIPELINE)
###############################################################################

@torch.no_grad()
def run_autoregressive_inference(
    model: GPTLightning,
    dataloader,
    device: torch.device,
    T_ctx: int,
    T_fut: int,
    max_context_window: Optional[int] = None,
    max_events: Optional[int] = None,
    use_stitcher: bool = False,
    use_freq_features: bool = True,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
) -> dict:
    """
    Autoregressive rollout evaluation.

    For each sample in the test set:
    1. Reconstruct full token sequence from (x, y).
    2. Split into context (first T_ctx tokens) and ground-truth future (next T_fut).
    3. Roll out T_fut tokens autoregressively.
    4. Flatten predicted and true future tokens to waveforms.
    5. Optionally apply the stitcher to the predicted waveform.

    Returns dict with per-event waveforms and theta (same format as
    teacher-forced output for downstream overlap computation).
    """
    model.eval()
    model.to(device)
    K = model.hparams.kernel_size

    hp_true_all, hc_true_all = [], []
    hp_pred_all, hc_pred_all = [], []
    theta_all = []
    count = 0

    for batch_idx, batch in enumerate(dataloader):
        x = batch["x"]       # [B, T, C, K]
        y = batch["y"]       # [B, T*K, C]
        theta = batch["theta"]  # [B, 4]

        B = x.shape[0]

        # Reconstruct full token sequence: [B, T+1, C, K]
        full_tokens = reconstruct_full_tokens(x, y)
        T_total = full_tokens.shape[1]

        # Check this batch has enough tokens
        T_needed = T_ctx + T_fut
        if T_total < T_needed:
            print(
                f"  [WARN] Skipping batch {batch_idx}: only {T_total} tokens "
                f"available but need {T_needed} (T_ctx={T_ctx} + T_fut={T_fut})"
            )
            continue

        # Split
        ctx = full_tokens[:, :T_ctx].to(device)             # [B, T_ctx, C, K]
        future_true = full_tokens[:, T_ctx:T_ctx + T_fut]   # [B, T_fut, C, K]
        theta_dev = theta.to(device)

        # Autoregressive rollout
        pred_tokens = autoregressive_rollout(
            model, ctx, theta_dev, T_fut,
            max_context_window=max_context_window,
            use_freq_features=use_freq_features,
            freq_keep_bins=freq_keep_bins,
            freq_log1p=freq_log1p,
        )  # [B, T_fut, C, K]  on device

        # Flatten to waveforms
        wf_true = tokens_to_waveform(future_true)       # [B, L, C]  CPU
        wf_pred = tokens_to_waveform(pred_tokens.cpu())  # [B, L, C]  CPU

        # Optional stitcher pass on predicted waveform
        if use_stitcher:
            wf_pred = apply_stitcher_to_waveform(model, wf_pred)

        wf_true_np = wf_true.float().numpy()
        wf_pred_np = wf_pred.float().numpy()
        theta_np = theta.float().numpy()

        for i in range(B):
            if max_events is not None and count >= max_events:
                break
            hp_true_all.append(wf_true_np[i, :, 0])
            hc_true_all.append(wf_true_np[i, :, 1])
            hp_pred_all.append(wf_pred_np[i, :, 0])
            hc_pred_all.append(wf_pred_np[i, :, 1])
            theta_all.append(theta_np[i])
            count += 1

        if max_events is not None and count >= max_events:
            break

    return {
        "hp_true": hp_true_all, "hc_true": hc_true_all,
        "hp_pred": hp_pred_all, "hc_pred": hc_pred_all,
        "theta": np.array(theta_all),
    }


###############################################################################
# §8  BATCH OVERLAP COMPUTATION
###############################################################################

def compute_all_overlaps(
    results: dict,
    fs: float,
    psd_type: str,
    psd_file: Optional[str],
    fmin: float,
    fmax: float,
) -> np.ndarray:
    """Compute maximised overlap for every event."""
    N = len(results["hp_true"])
    overlaps = np.zeros(N, dtype=np.float64)

    # Build PSD once using the length of the first waveform
    L = len(results["hp_true"][0])
    freqs = np.fft.rfftfreq(L, d=1.0 / fs)
    psd = load_psd(psd_type, psd_file, freqs, low_freq_cutoff=fmin)

    for i in range(N):
        h_true = build_complex_strain(results["hp_true"][i], results["hc_true"][i])
        h_pred = build_complex_strain(results["hp_pred"][i], results["hc_pred"][i])
        overlaps[i] = compute_overlap_maximised(
            h_true, h_pred, fs, psd, freqs, fmin, fmax
        )
        if (i + 1) % 500 == 0 or i == N - 1:
            print(
                f"  Overlap {i + 1}/{N}  "
                f"(running median={np.median(overlaps[:i+1]):.6f})"
            )

    return overlaps


###############################################################################
# §9  DERIVED PARAMETERS
###############################################################################

def derive_params(theta: np.ndarray) -> dict:
    """
    theta columns: [log(Mc), eta, s1z, s2z]
    Returns dict of 1-D arrays: logMc, eta, s1z, s2z, chi_eff, Mc, q
    """
    logMc, eta, s1z, s2z = theta[:, 0], theta[:, 1], theta[:, 2], theta[:, 3]
    Mc = np.exp(logMc)
    disc = np.clip(1.0 - 4.0 * eta, 0.0, None)
    q = (1.0 - 2.0 * eta - np.sqrt(disc)) / (2.0 * eta + 1e-30)
    q = np.clip(q, 0.0, 1.0)
    chi_eff = (s1z + q * s2z) / (1.0 + q)
    return dict(logMc=logMc, eta=eta, s1z=s1z, s2z=s2z,
                chi_eff=chi_eff, Mc=Mc, q=q)


###############################################################################
# §10  PLOTTING
###############################################################################

def plot_overlap_histogram(overlaps: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    lo = min(0.9, overlaps.min() - 0.005)
    ax.hist(overlaps, bins=80, range=(lo, 1.0),
            color="#2c7bb6", edgecolor="white", linewidth=0.4)
    ax.axvline(np.median(overlaps), ls="--", color="k", lw=1.2,
               label=f"median = {np.median(overlaps):.5f}")
    ax.set_xlabel("Maximised overlap $\\mathcal{O}$", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title("Overlap distribution", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_overlap_cdf(overlaps: np.ndarray, out_path: str):
    sorted_o = np.sort(overlaps)
    cdf = np.arange(1, len(sorted_o) + 1) / len(sorted_o)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sorted_o, cdf, color="#d7191c", lw=1.5)
    for thr in [0.99, 0.995, 0.999]:
        frac = np.mean(overlaps >= thr)
        ax.axvline(thr, ls=":", color="grey", lw=0.8)
        ax.text(thr, 0.05, f"{frac*100:.1f}%$\\geq${thr}", fontsize=7,
                rotation=90, va="bottom", ha="right", color="grey")
    ax.set_xlabel("Maximised overlap $\\mathcal{O}$", fontsize=12)
    ax.set_ylabel("CDF", fontsize=12)
    ax.set_title("Cumulative overlap distribution", fontsize=13)
    ax.set_xlim(min(0.9, sorted_o[0] - 0.005), 1.001)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_mismatch_scatter(params: dict, overlaps: np.ndarray, out_path: str):
    mismatch = 1.0 - overlaps
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(
        params["eta"], params["chi_eff"],
        c=np.log10(np.clip(mismatch, 1e-8, None)),
        s=6, cmap="inferno", rasterized=True,
    )
    cb = fig.colorbar(sc, ax=ax, pad=0.02)
    cb.set_label("log$_{10}$(1 − $\\mathcal{O}$)", fontsize=11)
    ax.set_xlabel("$\\eta$", fontsize=12)
    ax.set_ylabel("$\\chi_{\\rm eff}$", fontsize=12)
    ax.set_title("Mismatch over parameter space", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_mismatch_vs_chirpmass(params: dict, overlaps: np.ndarray, out_path: str):
    mismatch = 1.0 - overlaps
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(params["Mc"], mismatch, s=5, alpha=0.5,
               color="#2c7bb6", rasterized=True)
    ax.set_xlabel("Chirp mass $\\mathcal{M}_c$", fontsize=12)
    ax.set_ylabel("Mismatch $(1 - \\mathcal{O})$", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Mismatch vs chirp mass", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_mismatch_vs_q(params: dict, overlaps: np.ndarray, out_path: str):
    mismatch = 1.0 - overlaps
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.scatter(params["q"], mismatch, s=5, alpha=0.5,
               color="#fdae61", rasterized=True)
    ax.set_xlabel("Mass ratio $q$", fontsize=12)
    ax.set_ylabel("Mismatch $(1 - \\mathcal{O})$", fontsize=12)
    ax.set_yscale("log")
    ax.set_title("Mismatch vs mass ratio", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"  Saved {out_path}")


def plot_worst_k_waveforms(
    results: dict, overlaps: np.ndarray, k: int, out_dir: str,
    fs: float = 4096.0,
):
    """Plot the k worst-overlap waveforms with a time axis in milliseconds."""
    worst_idx = np.argsort(overlaps)[:k]
    for rank, idx in enumerate(worst_idx):
        hp_t, hp_p = results["hp_true"][idx], results["hp_pred"][idx]
        hc_t, hc_p = results["hc_true"][idx], results["hc_pred"][idx]
        L = len(hp_t)
        t_ms = np.arange(L) / fs * 1000.0  # time in ms

        fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(t_ms, hp_t, lw=0.6, color="k", label="true $h_+$")
        axes[0].plot(t_ms, hp_p, lw=0.6, color="#d7191c", alpha=0.8, label="pred $h_+$")
        axes[0].legend(fontsize=8, loc="upper left"); axes[0].set_ylabel("$h_+$")

        axes[1].plot(t_ms, hc_t, lw=0.6, color="k", label="true $h_\\times$")
        axes[1].plot(t_ms, hc_p, lw=0.6, color="#2c7bb6", alpha=0.8, label="pred $h_\\times$")
        axes[1].legend(fontsize=8, loc="upper left"); axes[1].set_ylabel("$h_\\times$")

        axes[2].plot(t_ms, hp_t - hp_p, lw=0.5, color="#d7191c", alpha=0.7, label="$\\Delta h_+$")
        axes[2].plot(t_ms, hc_t - hc_p, lw=0.5, color="#2c7bb6", alpha=0.7, label="$\\Delta h_\\times$")
        axes[2].legend(fontsize=8, loc="upper left")
        axes[2].set_ylabel("Residual"); axes[2].set_xlabel("Time [ms]")

        fig.suptitle(
            f"Worst #{rank+1}  (event {idx}, overlap={overlaps[idx]:.6f})", fontsize=12
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = os.path.join(out_dir, f"worst_{rank+1:02d}_event{idx}.png")
        fig.savefig(fname, dpi=150); plt.close(fig)
        print(f"  Saved {fname}")


###############################################################################
# §11  SUMMARY
###############################################################################

def build_summary(overlaps: np.ndarray) -> dict:
    return {
        "n_events": int(len(overlaps)),
        "mean_overlap": float(np.mean(overlaps)),
        "median_overlap": float(np.median(overlaps)),
        "min_overlap": float(np.min(overlaps)),
        "max_overlap": float(np.max(overlaps)),
        "std_overlap": float(np.std(overlaps)),
        "percentiles": {
            f"{p}%": float(np.percentile(overlaps, p))
            for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
        },
        "fraction_above": {
            str(thr): float(np.mean(overlaps >= thr))
            for thr in [0.90, 0.95, 0.99, 0.995, 0.999]
        },
        "mean_mismatch": float(np.mean(1.0 - overlaps)),
        "median_mismatch": float(np.median(1.0 - overlaps)),
    }


###############################################################################
# §12  CLI & MAIN
###############################################################################

def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate waveform surrogate via noise-weighted overlap.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # ---- paths ----
    p.add_argument("--ckpt", type=str, required=True,
                   help="Lightning checkpoint (.ckpt)")
    p.add_argument("--data_dir", type=str, required=True,
                   help="HDF5 waveform data directory")
    p.add_argument("--out_dir", type=str, default="results/overlap")

    # ---- mode ----
    p.add_argument("--mode", type=str, default="rollout",
                   choices=["teacher", "rollout"],
                   help="Evaluation mode")

    # ---- autoregressive context / future ----
    p.add_argument("--context_s", type=float, default=None,
                   help="Context duration in seconds (rollout mode)")
    p.add_argument("--future_s", type=float, default=None,
                   help="Future duration to predict in seconds (rollout mode)")
    p.add_argument("--context_tokens", type=int, default=None,
                   help="Context length in tokens (alternative to --context_s)")
    p.add_argument("--future_tokens", type=int, default=None,
                   help="Future length in tokens (alternative to --future_s)")
    p.add_argument("--max_context_window", type=int, default=None,
                   help="Sliding window size during rollout (tokens)")
    p.add_argument("--use_stitcher", action="store_true",
                   help="Apply stitcher to predicted waveform post-rollout")

    # ---- PSD ----
    p.add_argument("--psd", type=str, default="csv",
                   choices=["csv", "et_analytic", "flat"],
                   help="PSD type")
    p.add_argument("--psd_file", type=str, default=None,
                   help="CSV file with 'frequency','psd' columns")

    # ---- physics ----
    p.add_argument("--fs", type=float, default=4096.0,
                   help="Sample rate [Hz]")
    p.add_argument("--fmin", type=float, default=2.0,
                   help="Lower frequency bound [Hz]")
    p.add_argument("--fmax", type=float, default=1024.0,
                   help="Upper frequency bound [Hz]")

    # ---- data ----
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_files", type=int, default=None)
    p.add_argument("--n_samples_per_file", type=int, default=10000)
    p.add_argument("--kernel_size", type=int, default=None,
                   help="Override kernel_size (default: from checkpoint)")
    p.add_argument("--stride", type=int, default=None,
                   help="Override stride (default: same as kernel_size)")

    # ---- misc ----
    p.add_argument("--max_events", type=int, default=None,
                   help="Cap on test events (for quick runs)")
    p.add_argument("--worst_k", type=int, default=10)
    p.add_argument("--device", type=str,
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def seconds_to_tokens(seconds: float, fs: float, kernel_size: int, stride: int) -> int:
    """Convert a duration in seconds to number of tokens."""
    n_samples = seconds * fs
    # Each token spans `stride` new samples (non-overlapping when stride == K)
    return max(1, int(round(n_samples / stride)))


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    # ==================================================================
    # 1. Load model
    # ==================================================================
    print(f"Loading checkpoint: {args.ckpt}")
    model = GPTLightning.load_from_checkpoint(args.ckpt, map_location="cpu")
    model.to(device).eval()
    K = model.hparams.kernel_size
    C = model.hparams.in_channels
    print(f"  kernel_size={K}, in_channels={C}, d_model={model.hparams.d_model}, "
          f"num_enc_layers={model.hparams.num_enc_layers}")

    kernel_size = args.kernel_size or K
    stride = args.stride or kernel_size

    # ==================================================================
    # 2. Build test dataloader
    # ==================================================================
    dm = GWDataModule(
        data_dir=args.data_dir,
        n_files=args.n_files,
        n_samples_per_file=args.n_samples_per_file,
        kernel_size=kernel_size,
        stride=stride,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.setup(stage="test")
    test_loader = dm.test_dataloader()
    print(f"  Test set size: {len(dm.test_dataset)}")

    # ==================================================================
    # 3. Inference
    # ==================================================================
    t0 = time.time()

    if args.mode == "teacher":
        print("\n--- Teacher-forced evaluation ---")
        results = run_teacher_forced_inference(
            model, test_loader, device, max_events=args.max_events,
        )

    elif args.mode == "rollout":
        print("\n--- Autoregressive rollout evaluation ---")

        # Resolve context / future in tokens
        if args.context_tokens is not None:
            T_ctx = args.context_tokens
        elif args.context_s is not None:
            T_ctx = seconds_to_tokens(args.context_s, args.fs, kernel_size, stride)
        else:
            T_ctx = None  # set from data below

        if args.future_tokens is not None:
            T_fut = args.future_tokens
        elif args.future_s is not None:
            T_fut = seconds_to_tokens(args.future_s, args.fs, kernel_size, stride)
        else:
            T_fut = None

        # Peek at first batch to determine total token count
        _peek = next(iter(test_loader))
        T_available = _peek["x"].shape[1] + 1  # full token count = T + 1
        print(f"  Tokens per waveform: {T_available}")
        print(f"  Token duration: {stride / args.fs * 1000:.2f} ms "
              f"({stride} samples at {args.fs} Hz)")

        if T_ctx is None:
            T_ctx = T_available // 2
        if T_fut is None:
            T_fut = T_available - T_ctx
        # Clamp
        if T_ctx + T_fut > T_available:
            T_fut = T_available - T_ctx
            print(f"  [NOTE] Clamped T_fut to {T_fut} (limited by waveform length)")

        ctx_s = T_ctx * stride / args.fs
        fut_s = T_fut * stride / args.fs
        print(f"  Context:  {T_ctx} tokens = {ctx_s:.4f} s = {T_ctx * stride} samples")
        print(f"  Future:   {T_fut} tokens = {fut_s:.4f} s = {T_fut * stride} samples")

        # Uni-modal (time-only) models have no frequency_token_embed
        use_freq = getattr(model.gpt, "frequency_token_embed", None) is not None
        results = run_autoregressive_inference(
            model, test_loader, device,
            T_ctx=T_ctx,
            T_fut=T_fut,
            max_context_window=args.max_context_window,
            max_events=args.max_events,
            use_stitcher=args.use_stitcher,
            use_freq_features=use_freq,
        )

    t_infer = time.time() - t0
    n_events = len(results["hp_true"])
    if n_events == 0:
        print("ERROR: No events produced. Check data and context/future settings.")
        sys.exit(1)
    print(f"  Inference done: {n_events} events in {t_infer:.1f}s "
          f"({n_events / max(t_infer, 1e-6):.0f} events/s)")

    # ==================================================================
    # 4. Compute overlaps
    # ==================================================================
    print(f"\nComputing overlaps (PSD={args.psd}, fmin={args.fmin}, fmax={args.fmax}) ...")
    t0_ov = time.time()
    overlaps = compute_all_overlaps(
        results, fs=args.fs,
        psd_type=args.psd, psd_file=args.psd_file,
        fmin=args.fmin, fmax=args.fmax,
    )
    t_overlap = time.time() - t0_ov
    print(f"  Overlap computation: {t_overlap:.1f}s")

    # ==================================================================
    # 5. Derived parameters & save
    # ==================================================================
    params = derive_params(results["theta"])
    print(f"\nSaving to {args.out_dir}/ ...")

    np.save(os.path.join(args.out_dir, "overlaps.npy"), overlaps)
    np.save(os.path.join(args.out_dir, "params.npy"), results["theta"])

    csv_path = os.path.join(args.out_dir, "params.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["logMc", "eta", "s1z", "s2z", "chi_eff", "q", "Mc",
                     "overlap", "mismatch"])
        for i in range(n_events):
            w.writerow([
                f"{params['logMc'][i]:.6f}", f"{params['eta'][i]:.6f}",
                f"{params['s1z'][i]:.6f}", f"{params['s2z'][i]:.6f}",
                f"{params['chi_eff'][i]:.6f}", f"{params['q'][i]:.6f}",
                f"{params['Mc'][i]:.6f}",
                f"{overlaps[i]:.8f}", f"{1.0 - overlaps[i]:.8e}",
            ])

    summary = build_summary(overlaps)
    summary["mode"] = args.mode
    summary["inference_time_s"] = round(t_infer, 2)
    summary["overlap_time_s"] = round(t_overlap, 2)
    summary["throughput_events_per_s"] = round(n_events / max(t_infer, 1e-6), 1)
    if args.mode == "rollout":
        summary["context_tokens"] = T_ctx
        summary["future_tokens"] = T_fut
        summary["context_seconds"] = round(T_ctx * stride / args.fs, 6)
        summary["future_seconds"] = round(T_fut * stride / args.fs, 6)
    summary["args"] = {k: str(v) for k, v in vars(args).items()}
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print(f"  OVERLAP SUMMARY  ({args.mode} mode)")
    print("=" * 60)
    print(f"  Events:          {summary['n_events']}")
    print(f"  Mean overlap:    {summary['mean_overlap']:.6f}")
    print(f"  Median overlap:  {summary['median_overlap']:.6f}")
    print(f"  Min overlap:     {summary['min_overlap']:.6f}")
    print(f"  Mean mismatch:   {summary['mean_mismatch']:.6e}")
    print(f"  Median mismatch: {summary['median_mismatch']:.6e}")
    for thr, frac in summary["fraction_above"].items():
        print(f"  Fraction >= {thr}: {frac*100:.1f}%")
    print("=" * 60)

    # ==================================================================
    # 6. Plots
    # ==================================================================
    print("\nGenerating figures ...")
    plot_overlap_histogram(overlaps, os.path.join(args.out_dir, "overlap_hist.png"))
    plot_overlap_cdf(overlaps, os.path.join(args.out_dir, "overlap_cdf.png"))
    plot_mismatch_scatter(params, overlaps,
                          os.path.join(args.out_dir, "mismatch_eta_chieff.png"))
    plot_mismatch_vs_chirpmass(params, overlaps,
                               os.path.join(args.out_dir, "mismatch_vs_chirpmass.png"))
    plot_mismatch_vs_q(params, overlaps,
                       os.path.join(args.out_dir, "mismatch_vs_q.png"))
    plot_worst_k_waveforms(results, overlaps, args.worst_k, args.out_dir, fs=args.fs)

    print("\nDone.")


if __name__ == "__main__":
    main()