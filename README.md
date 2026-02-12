# GW-Autoregressor

Autoregressive surrogate model for **gravitational-wave (GW) waveforms**. A causal transformer is trained to predict merger-centered strain windows token-by-token, conditioned on binary black hole parameters. Evaluation uses **noise-weighted overlap** (teacher-forced or autoregressive rollout), suitable for Einstein Telescope–style sensitivity.

**Branch:** `main`

---

## Data & Data Handling

- **Source:** HDF5 files in a single directory (e.g. `merger_windows_*.h5`). Each file contains:
  - **Waveforms:** `waveforms/h_plus`, `waveforms/h_cross` — strain time series (h₊, h×) per sample.
  - **Parameters:** `params/mass1`, `params/mass2`, `params/spin1z`, `params/spin2z` — converted to a conditioning vector **θ** = [log(chirp mass), η, s1z, s2z].

- **Tokenization:** Waveforms are scaled (e.g. ×10²⁴), then unfolded into **non-overlapping time tokens** of length `kernel_size` (e.g. 64) with the same `stride`. Each sample yields input tokens **x** [T, C, K], next-token targets **y** [T×K, C], and per-token **frequency features** (low-frequency FFT magnitude bins, optional log1p) for the model.

- **Dataset:** `MergerWindowDataset` opens HDF5 **lazily** (one handle per DataLoader worker, SWMR) and preloads only the small θ arrays. This keeps memory low while avoiding open/close on every `__getitem__`. HDF5 is not fork-safe; a `worker_init_fn` clears inherited handles so each worker opens files in its own process.

- **DataModule:** `GWDataModule` builds the dataset and splits it **80/10/10** train/val/test, and provides DataLoaders with optional `num_workers`, `pin_memory`, and `persistent_workers`.

---

## Model

- **Architecture:** GPT-style **causal transformer** with:
  - **Token embedding:** Two branches over each token (length K): (1) **time-domain** — dilated 1D CNN + attention pooling over K; (2) **frequency-domain** — same structure on per-token FFT magnitude features. The two are fused (e.g. add or cross-attention) into a single sequence of token embeddings.
  - **Sequence model:** Stack of **RoPE** transformer encoder layers (pre-norm, Flash Attention–friendly: causal masking via `is_causal`, no explicit mask). Optional **conditioning** on θ through an MLP and **AdaLayerNorm** (FiLM-style) in each layer.
  - **Output:** Linear head predicts the next token’s waveform slice → [B, T×K, C]. An optional **causal 1D stitcher** smooths predictions along the sample axis to reduce token-boundary artifacts.

- **Training:** PyTorch Lightning; loss is **energy-weighted** (e.g. MSE or L1 over token residuals, weighted by per-token RMS). Config-driven via `LightningCLI` (see `configs/train.yaml`).

- **Evaluation:** `eval_and_plot.py` loads a checkpoint and either (1) **teacher-forced** evaluation or (2) **autoregressive rollout** (context window in seconds/tokens, then generate future tokens). It computes maximised noise-weighted overlap (time and phase), supports PSD from CSV or analytic ET-D, and writes overlap statistics and plots (histograms, CDF, mismatch vs parameters, worst-k waveforms).

---

## Quick Start

- **Train:**  
  `python main.py fit --config configs/train.yaml`  
  (set `data.init_args.data_dir` to your HDF5 directory.)

- **Evaluate (e.g. autoregressive overlap):**  
  `python eval_and_plot.py --ckpt <path>.ckpt --data_dir <path> --mode rollout --context_s 0.5 --future_s 0.25 --psd et_analytic --out_dir results/rollout`

---

## Repo Layout

- `main.py` — Lightning CLI entrypoint (fit/validate/test).
- `eval_and_plot.py` — Overlap evaluation and plotting (teacher + rollout).
- `configs/train.yaml` — Training and data config.
- `datasets/data_handling.py` — `MergerWindowDataset`, `GWDataModule`, worker init.
- `models/models.py` — GPT, token embeddings, RoPE, stitcher.
- `models/lightning_module.py` — `GPTLightning` (loss, optimizers, logging).

Data (HDF5) and checkpoints are not in the repo; point `data_dir` and `--ckpt` to your own paths.
