# GW-Autoregressor

Autoregressive surrogate model for **gravitational-wave (GW) waveforms**. A causal transformer is trained to predict merger-centered strain windows token-by-token, conditioned on binary black hole parameters. Evaluation uses **noise-weighted overlap** (teacher-forced or autoregressive rollout), suitable for Einstein Telescope–style sensitivity.

**Branch:** `dev`

---

## Data & Data Handling

- **Source:** HDF5 files in a single directory (any `*.h5`). Each file contains:
  - **Waveforms:** `waveforms/h_plus`, `waveforms/h_cross` — strain time series (h₊, h×) per sample.
  - **Parameters:** `params/mass1`, `params/mass2`, `params/spin1z`, `params/spin2z` — converted to a conditioning vector **θ** = [log(chirp mass), η, s1z, s2z].

- **Tokenization:** Waveforms are scaled by 10²⁴, then unfolded into **non-overlapping time tokens** of length `kernel_size` with the same `stride` (e.g. 64). Optionally, each waveform is truncated to `max_samples` time steps (e.g. 57500). Each sample yields:
  - **x** [T, C, K] — input time tokens (C=2 for h₊, h×).
  - **y** [T×K, C] — next-token targets (flattened).
  - **x_freq** [T, C, F] — per-token **frequency features**: low-frequency rFFT magnitude bins (first `freq_keep_bins`, e.g. 4), with optional `log1p` and normalization (`freq_norm`: `"none"` | `"mean"` | `"l2"`). Computed via `freq_features_from_tokens()` so dataset, training (scheduled sampling), and inference rollout use the same recipe and avoid leakage.

- **Dataset:** `MergerWindowDataset` opens HDF5 **lazily** (one handle per DataLoader worker, SWMR) and **preloads only θ** from all files (tiny memory). This avoids loading full waveforms into RAM and avoids open/close on every `__getitem__`. HDF5 is not fork-safe; `worker_init_fn` clears inherited handles so each worker opens files in its own process. Config: `n_files`, `n_samples_per_file`, `kernel_size`, `stride`, `freq_keep_bins`, `freq_log1p`, `freq_norm`, `max_samples`.

- **DataModule:** `GWDataModule` builds the dataset and splits it **80/10/10** train/val/test, and provides DataLoaders with `num_workers`, `pin_memory`, `persistent_workers`, and `worker_init_fn`.

---

## Model

- **Architecture:** GPT-style **causal transformer** (`models/models.py`):
  - **Token embedding:** Two branches per token (length K):
    - **Time:** `TokenEmbedding` — dilated 1D CNN over K + attention pooling → [B, T, d_model].
    - **Frequency:** `FreqMLPEmbed` (or legacy conv) on per-token FFT features [B, T, C, F] → [B, T, d_model].
  - **Fusion:** Time and frequency embeddings are combined via `fusion_type`: `"concat"` (default), `"add"`, or `"cross_attention"` → single sequence [B, T, d_model].
  - **Conditioning:** **θ** [B, 4] is mapped by an MLP to `cond_dim` and injected into every transformer layer via **AdaLayerNorm** (FiLM-style).
  - **Sequence:** Stack of **RoPE** transformer encoder layers (pre-norm; causal masking via `is_causal`, no explicit mask, so Flash Attention can be used). Optional **KV-cache** for fast autoregressive decoding.
  - **Output:** Linear head → next-token waveform slice [B, T×K, C]. Optional **CausalStitcher1D** smooths along the sample axis to reduce token-boundary artifacts.

- **Training** (`models/lightning_module.py`): PyTorch Lightning (`GPTLightning`). Loss is **energy-weighted** over token residuals (per-token RMS weight; base loss: `time_loss` = `"mse"` | `"l1"` | `"log_cosh"`). Optimizer: AdamW; optional **CosineAnnealingWarmRestarts** LR scheduler. Config-driven via `LightningCLI` and `configs/train.yaml` (e.g. `bf16-mixed`, `accumulate_grad_batches`, EarlyStopping, ModelCheckpoint).

- **Scheduled sampling (optional):** When `ss_enabled: true`, training adds short **autoregressive unrolls** on top of teacher-forced loss: from a random start (optionally in the last `ss_focus_fraction` of the sequence), the model unrolls `ss_unroll_steps` tokens, at each step feeding back its own prediction with probability *p* (linearly annealed from `ss_p_start` to `ss_p_end` over `ss_warmup_steps`). Frequency features for predicted tokens are recomputed with `freq_features_from_tokens` (same as dataset and rollout). Optional `ss_use_cache` uses the model’s KV-cache during unroll; `ss_detach_pred` detaches fed-back predictions for stability.

- **Evaluation:** `eval_and_plot.py` — teacher-forced or **autoregressive rollout** (context window in seconds, then generate future tokens); maximised noise-weighted overlap, PSD from CSV or analytic ET-D; overlap statistics and plots. `autoregressive_rollout.py` — load checkpoint, run rollout on test data, optional PyCBC match/mismatch and comparison plot.

---

## Quick Start

- **Train:**  
  `python main.py fit --config configs/train.yaml`  
  Set `data.init_args.data_dir` to your HDF5 directory; tune `data.init_args.n_files`, `max_samples`, `batch_size`, and model/data `freq_*` so they match.

- **Evaluate (e.g. autoregressive overlap):**  
  `python eval_and_plot.py --ckpt <path>.ckpt --data_dir <path> --mode rollout --context_s 0.5 --future_s 0.25 --psd et_analytic --out_dir results/rollout`

---

## Repo Layout

- `main.py` — Lightning CLI entrypoint (fit/validate/test).
- `eval_and_plot.py` — Overlap evaluation and plotting (teacher + rollout).
- `autoregressive_rollout.py` — Rollout script with optional PyCBC match and plots.
- `configs/train.yaml` — Training and data config (model, data, trainer).
- `datasets/data_handling.py` — `freq_features_from_tokens`, `MergerWindowDataset`, `GWDataModule`, `_worker_init`.
- `models/models.py` — GPT, TokenEmbedding, FreqMLPEmbed, RoPE, AdaLayerNorm, CausalStitcher1D, KV-cache.
- `models/lightning_module.py` — `GPTLightning` (loss, optimizers, scheduled sampling, logging).

Data (HDF5) and checkpoints are not in the repo; point `data_dir` and `--ckpt` to your own paths.
