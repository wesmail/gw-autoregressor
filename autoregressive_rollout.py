#!/usr/bin/env python3
"""
Autoregressive rollout script: load a trained checkpoint, run teacher-forced or
free (autoregressive) rollout on a test sample, and save a comparison plot.
Optionally computes PyCBC match/mismatch (overlap_max) and prints it on the plot.

Usage:
  python autoregressive_rollout.py --ckpt <path>.ckpt --data_dir <path> [options]
"""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from datasets.data_handling import GWDataModule
from models.lightning_module import GPTLightning

# Optional PyCBC for match/mismatch
try:
    from pycbc.types import TimeSeries
    from pycbc.filter.matchedfilter import match
    from pycbc.psd import analytical as psd_analytical
    PYCBC_AVAILABLE = True
except ImportError:
    PYCBC_AVAILABLE = False


def compute_match_mismatch(
    y_true_future: np.ndarray,
    y_pred_future: np.ndarray,
    dt: float,
    f_low: float,
    psd_builder,
    psd_kwargs: dict | None = None,
) -> dict[str, Any]:
    """
    Compute PSD-weighted match (overlap_max) and mismatch between true and predicted
    future waveforms using PyCBC.

    y_true_future: [L, 2] (h+, h×) or [L, C] (first 2 cols used)
    y_pred_future: [L, 2]
    dt: sample spacing in seconds
    f_low: low-frequency cutoff in Hz
    psd_builder: callable(length, delta_f, low_freq_cutoff, **kwargs) -> FrequencySeries
    psd_kwargs: optional kwargs passed to psd_builder

    Returns:
      dict with match_plus, match_cross, match_avg, mismatch_avg, etc.
    """
    psd_kwargs = psd_kwargs or {}

    yt = np.asarray(y_true_future[:, :2], dtype=np.float64)
    yp = np.asarray(y_pred_future[:, :2], dtype=np.float64)
    assert yt.shape == yp.shape and yt.ndim == 2 and yt.shape[1] == 2

    N = yt.shape[0]
    df = 1.0 / (N * dt)
    nfreq = N // 2 + 1

    psd = psd_builder(length=nfreq, delta_f=df, low_freq_cutoff=f_low, **psd_kwargs)

    ht_p = TimeSeries(yt[:, 0], delta_t=dt)
    hp_p = TimeSeries(yp[:, 0], delta_t=dt)
    ht_x = TimeSeries(yt[:, 1], delta_t=dt)
    hp_x = TimeSeries(yp[:, 1], delta_t=dt)

    m_plus, idx_plus = match(ht_p, hp_p, psd=psd, low_frequency_cutoff=f_low)
    m_cross, idx_cross = match(ht_x, hp_x, psd=psd, low_frequency_cutoff=f_low)
    m_avg = 0.5 * (m_plus + m_cross)

    return {
        "match_plus": float(m_plus),
        "mismatch_plus": float(1.0 - m_plus),
        "shift_plus_sec": float(idx_plus * dt),
        "match_cross": float(m_cross),
        "mismatch_cross": float(1.0 - m_cross),
        "shift_cross_sec": float(idx_cross * dt),
        "match_avg": float(m_avg),
        "mismatch_avg": float(1.0 - m_avg),
        "N_samples": int(N),
        "dt": float(dt),
        "df": float(df),
    }


def default_psd_builder(length: int, delta_f: float, low_freq_cutoff: float, **kwargs: Any):
    """Default aLIGO ZeroDetHighPower PSD for match (PyCBC)."""
    return psd_analytical.EinsteinTelescopeP1600143(length, delta_f, low_freq_cutoff)


def freq_features_from_tokens(
    x: torch.Tensor,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
) -> torch.Tensor:
    """
    Per-token FFT magnitude, matching data_handling.MergerWindowDataset._freq_features.
    x: [B, T, C, K] → [B, T, C, F]
    """
    mag = torch.fft.rfft(x, dim=-1).abs()
    if freq_log1p:
        mag = torch.log1p(mag)
    Fkeep = min(freq_keep_bins, mag.shape[-1])
    return mag[..., :Fkeep]


def flatten_tokens_to_samples(tokens_1B_TCK: torch.Tensor) -> torch.Tensor:
    """
    tokens: [1, T, C, K] -> samples: [T*K, C]
    """
    assert tokens_1B_TCK.ndim == 4 and tokens_1B_TCK.shape[0] == 1
    T, C, K = tokens_1B_TCK.shape[1], tokens_1B_TCK.shape[2], tokens_1B_TCK.shape[3]
    return tokens_1B_TCK.squeeze(0).permute(0, 2, 1).reshape(T * K, C)


def rollout(
    model,
    data,
    item: int,
    context_fraction: float,
    future_fraction: float,
    device,
    kernel_size: int = 64,
    freq_keep_bins: int = 8,
    freq_log1p: bool = True,
    mode: str = "free",
    max_context_tokens: int | None = None,
):
    """
    Autoregressive rollout on token space.
    Aligned with MergerWindowDataset (x_freq from per-token FFT) and GPT forward(x_time, x_freq, ...).

    data["x"] : [B, T_full, C, K]  (true tokens)
    data["y"] : [B, T_full*K, C]   (shifted-by-1 samples)
    data["theta"] : [B, theta_dim]

    Returns:
      true_full_samples: [T_full*K, C]
      true_future_samples: [L_future, C]
      pred_future_samples: [L_future, C]
      L_context (in samples)
      L_future (in samples)
    """
    x_full = data["x"].to(device)
    theta = data["theta"].to(device)

    B, T_full, C, K = x_full.shape
    assert K == kernel_size, f"Expected K==kernel_size ({kernel_size}), got K={K}"
    assert 0 <= context_fraction <= 1, "context_fraction must be in [0,1]"
    assert 0 <= future_fraction <= 1, "future_fraction must be in [0,1]"
    assert mode in ("free", "teacher"), "mode must be 'free' or 'teacher'"

    # Multi-modal: model expects x_freq (from frequency_token_embed). Uni-modal: pass None.
    use_freq = getattr(model, "frequency_token_embed", None) is not None

    T_context = max(1, int(T_full * context_fraction))
    T_remain = T_full - T_context
    n_future_tokens = max(1, int(T_full * future_fraction))
    n_future_tokens = min(n_future_tokens, max(1, T_remain))

    L_context = T_context * K
    L_future = n_future_tokens * K

    x = x_full[item : item + 1, :T_context].clone()
    theta_i = theta[item : item + 1]
    gt_future_tokens = x_full[item : item + 1, T_context : T_context + n_future_tokens].clone()

    pred_list = []
    for i in range(n_future_tokens):
        if (max_context_tokens is not None) and (x.shape[1] > max_context_tokens):
            x = x[:, -max_context_tokens:, :, :].contiguous()

        if use_freq:
            x_freq = freq_features_from_tokens(x, freq_keep_bins, freq_log1p)
        else:
            x_freq = None

        with torch.no_grad():
            if x_freq is not None:
                out = model(x, x_freq, is_causal=True, theta=theta_i)
            else:
                out = model(x, is_causal=True, theta=theta_i)

        next_samples = out[:, -K:, :]
        pred_list.append(next_samples.squeeze(0).cpu())

        if mode == "teacher":
            x = torch.cat([x, gt_future_tokens[:, i : i + 1]], dim=1)
        else:
            next_token = next_samples.permute(0, 2, 1).unsqueeze(1)
            x = torch.cat([x, next_token], dim=1)

    pred_future_samples = torch.cat(pred_list, dim=0).numpy()
    true_future_samples = flatten_tokens_to_samples(gt_future_tokens).cpu().numpy()
    true_full_samples = flatten_tokens_to_samples(x_full[item : item + 1]).cpu().numpy()

    L = min(len(pred_future_samples), len(true_future_samples))
    pred_future_samples = pred_future_samples[:L]
    true_future_samples = true_future_samples[:L]

    return true_full_samples, true_future_samples, pred_future_samples, L_context, L


def main():
    p = argparse.ArgumentParser(
        description="Run autoregressive rollout and save plot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ckpt", type=str, required=True, help="Lightning checkpoint (.ckpt)")
    p.add_argument("--data_dir", type=str, required=True, help="HDF5 data directory")
    p.add_argument("--n_files", type=int, default=None, help="Limit number of HDF5 files")
    p.add_argument("--n_samples_per_file", type=int, default=10000)
    p.add_argument("--item", type=int, default=None, help="Batch index (default: random)")
    p.add_argument("--context_fraction", type=float, default=0.85, help="Fraction of waveform as context (0–1)")
    p.add_argument("--future_fraction", type=float, default=0.05, help="Fraction of waveform to predict (0–1)")
    p.add_argument("--mode", type=str, default="free", choices=["free", "teacher"],
                   help="free = autoregressive; teacher = teacher-forced")
    p.add_argument("--max_context_tokens", type=int, default=None,
                   help="Sliding context window size (optional)")
    p.add_argument("--kernel_size", type=int, default=None,
                   help="Token size (default: from checkpoint)")
    p.add_argument("--out", type=str, default="rollout.png", help="Output plot path")
    p.add_argument("--device", type=str, default=None,
                   help="Device (default: cuda if available)")
    p.add_argument("--xlim_left", type=float, default=None, help="Plot x-axis left limit (samples)")
    p.add_argument("--xlim_right", type=float, default=None, help="Plot x-axis right limit (samples)")
    p.add_argument("--fs", type=float, default=4096.0, help="Sample rate [Hz] for match/mismatch")
    p.add_argument("--f_low", type=float, default=20.0, help="Low-frequency cutoff [Hz] for PSD/match")
    p.add_argument("--no_match", action="store_true", help="Skip PyCBC match/mismatch computation")
    p.add_argument("--freq_keep_bins", type=int, default=4, help="Frequency bins for x_freq (match MergerWindowDataset)")
    p.add_argument("--freq_norm", type=str, default="mean", help="Frequency normalization for x_freq (match MergerWindowDataset)")
    p.add_argument("--no_freq_log1p", action="store_true", help="Disable log1p on FFT mags (dataset uses log1p by default)")
    p.add_argument("--max_samples", type=int, default=58000, help="Maximum number of samples to use from dataset")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Load model (Lightning wrapper -> raw GPT)
    print(f"Loading checkpoint: {args.ckpt}")
    lightning = GPTLightning.load_from_checkpoint(args.ckpt, map_location="cpu")
    model = lightning.gpt.to(device)
    model.eval()
    K = getattr(lightning.hparams, "kernel_size", args.kernel_size)
    if K is None:
        K = getattr(model, "kernel_size", 64)
    if args.kernel_size is not None:
        K = args.kernel_size
    print(f"  kernel_size={K}")

    # Data
    dm = GWDataModule(
        data_dir=args.data_dir,
        n_files=args.n_files,
        n_samples_per_file=args.n_samples_per_file,
        kernel_size=K,
        stride=K,
        freq_keep_bins=args.freq_keep_bins,
        freq_norm=args.freq_norm,
        batch_size=32,
        num_workers=0,
        max_samples=args.max_samples,
    )
    dm.setup("test")
    test_loader = dm.test_dataloader()
    data = next(iter(test_loader))

    item = args.item
    if item is None:
        item = int(np.random.choice(data["x"].shape[0]))
        print(f"  Random item: {item}")

    print(f"  Context fraction: {args.context_fraction}, future fraction: {args.future_fraction}, mode: {args.mode}")

    y_true_full, y_true_future, y_pred_rollout, L_ctx, L_fut = rollout(
        model=model,
        data=data,
        item=item,
        context_fraction=args.context_fraction,
        future_fraction=args.future_fraction,
        device=device,
        kernel_size=K,
        freq_keep_bins=args.freq_keep_bins,
        freq_log1p=not args.no_freq_log1p,
        mode=args.mode,
        max_context_tokens=args.max_context_tokens,
    )

    # Match/mismatch (PyCBC)
    match_result = None
    if not args.no_match and PYCBC_AVAILABLE:
        dt = 1.0 / args.fs
        try:
            match_result = compute_match_mismatch(
                y_true_future,
                y_pred_rollout,
                dt=dt,
                f_low=args.f_low,
                psd_builder=default_psd_builder,
            )
            print(f"  overlap_max (match_avg) = {match_result['match_avg']:.6f}")
            print(f"  mismatch_avg           = {match_result['mismatch_avg']:.6e}")
        except Exception as e:
            print(f"  [WARN] Match computation failed: {e}")
            match_result = None
    elif args.no_match:
        print("  Skipping match (--no_match)")
    else:
        print("  Skipping match (PyCBC not available)")

    # Plot
    fig, ax = plt.subplots(2, 1, figsize=(10, 4), sharex=True)
    t_full = np.arange(len(y_true_full))
    t_fut = np.arange(L_ctx, L_ctx + len(y_pred_rollout))

    ax[0].plot(t_full, y_true_full[:, 0], color="gray", lw=0.6, alpha=0.4, label="True (full)")
    ax[0].axvline(L_ctx, color="k", ls="--", alpha=0.7, label="Context end")
    ax[0].plot(t_fut, y_true_future[:, 0], lw=0.6, color="tab:orange", label="True future")
    ax[0].scatter(t_fut, y_pred_rollout[:, 0], color="tab:blue", marker="x", s=1, alpha=0.5, label=f"Pred future ({args.mode})")
    ax[0].set_ylabel("h₊")
    ax[0].legend(loc="upper right")
    ax[0].grid(alpha=0.3)
    if args.xlim_left is not None or args.xlim_right is not None:
        ax[0].set_xlim(left=args.xlim_left, right=args.xlim_right)

    ax[1].plot(t_full, y_true_full[:, 1], color="gray", lw=0.6, alpha=0.4)
    ax[1].axvline(L_ctx, color="k", ls="--", alpha=0.7)
    ax[1].plot(t_fut, y_true_future[:, 1], lw=0.6, color="tab:orange")
    ax[1].scatter(t_fut, y_pred_rollout[:, 1], color="tab:blue", marker="x", s=1, alpha=0.5, label="Pred future (rollout)")
    ax[1].set_ylabel("h×")
    ax[1].set_xlabel("Time samples")
    ax[1].grid(alpha=0.3)
    if args.xlim_left is not None or args.xlim_right is not None:
        ax[1].set_xlim(left=args.xlim_left, right=args.xlim_right)

    title = (
        f"{args.mode.capitalize()} rollout: {args.context_fraction*100:.0f}% context → "
        f"{args.future_fraction*100:.0f}% future (item {item})"
    )
    if match_result is not None:
        title += f"\noverlap_max = {match_result['match_avg']:.6f}  |  mismatch = {match_result['mismatch_avg']:.6e}"
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(args.out, dpi=600)
    plt.close()
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
