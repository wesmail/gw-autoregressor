"""
PyTorch Lightning module for training the GPT model (time-domain only) with
time-domain loss (MSE/L1) and multi-resolution STFT loss.

OPTIMIZATIONS:
- Flash Attention enabled (via is_causal flag in model)
- Optional torch.compile() for additional speedup
- Flash Attention status logging at init

FEATURES:
- Scheduled sampling with configurable annealing (behind flag)
- Optional KV-cache usage during scheduled sampling unroll
"""
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .models import GPT, print_flash_attention_status, compile_model
from datasets.data_handling import freq_features_from_tokens

# =============================================================================
# Log-Cosh Loss
# =============================================================================
import torch.nn as nn
class LogCoshLoss(nn.Module):
    def forward(self, x, y):  # x, y: [B, T, F]
        diff = x - y
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
    

class GPTLightning(LightningModule):
    """
    Lightning module for time-only GPT: takes x (time tokens) and y (time target),
    returns predictions and computes time + STFT loss.
    """

    def __init__(
        self,
        # Model (GPT) args — must match data kernel_size
        in_channels: int = 3,
        kernel_size: int = 16,
        d_model: int = 128,
        num_heads: int = 2,
        num_enc_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
        dim_feedforward_multiplier: int = 4,
        # Token causal CNN embedding
        token_cnn_kernel: int = 7,
        token_cnn_layers: int = 4,
        token_cnn_dilation_growth: int = 2,
        token_cnn_dropout: float = 0.0,
        # Post-head stitcher (smooths predictions over sample axis)
        use_stitcher: bool = True,
        stitcher_hidden: int = 64,
        stitcher_kernel: int = 9,
        stitcher_layers: int = 4,
        stitcher_dropout: float = 0.0,
        # Loss
        time_loss: str = "l1",
        fusion_type: str = "cross_attention",
        # frequency branch (must match data)
        freq_embed_type: str = "mlp",   # "mlp" or "conv" (legacy)
        freq_keep_bins: int = 8,
        freq_norm: str = "none",        # "none" | "mean" | "l2" — must match dataset
        freq_log1p: bool = True,
        # conditioning
        theta_dim: int = 0,
        cond_dim: int = 128,
        cond_hidden: int = 256,
        lr: float = 1e-4,
        # Optional LR scheduler (flat args for CLI; omit scheduler or set null = no scheduler)
        scheduler: Optional[str] = None,
        scheduler_T_0: int = 5,
        scheduler_T_mult: int = 2,
        scheduler_eta_min: float = 1e-6,
        lr_scheduler_interval: str = "epoch",
        lr_scheduler_frequency: int = 1,
        lr_scheduler_monitor: Optional[str] = None,
        # NEW: torch.compile() options
        use_torch_compile: bool = False,
        compile_mode: str = "reduce-overhead",
        # =============================================
        # NEW: Scheduled sampling config (flat args)
        # =============================================
        ss_enabled: bool = False,         # master switch
        ss_warmup_steps: int = 2000,      # linear anneal from p_start to p_end
        ss_p_start: float = 0.0,          # initial probability of using prediction
        ss_p_end: float = 0.2,            # final probability
        ss_unroll_steps: int = 2,         # number of autoregressive unroll steps
        ss_detach_pred: bool = True,      # detach fed-back predictions (stable training)
        ss_use_cache: bool = False,       # use KV-cache during unroll (optional speedup)
        ss_focus_fraction: float = 1.0,   # sample start from last X% of tokens (1.0 = full range)
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if time_loss not in ("mse", "l1", "log_cosh"):
            raise ValueError("time_loss must be 'mse', 'l1', or 'log_cosh'")
        self.time_loss = time_loss

        # Print Flash Attention status at initialization
        if torch.cuda.is_available():
            print_flash_attention_status()

        self.gpt = GPT(
            in_channels=in_channels,
            kernel_size=kernel_size,
            d_model=d_model,
            num_heads=num_heads,
            num_enc_layers=num_enc_layers,
            dropout=dropout,
            max_len=max_len,
            dim_feedforward_multiplier=dim_feedforward_multiplier,
            token_cnn_kernel=token_cnn_kernel,
            token_cnn_layers=token_cnn_layers,
            token_cnn_dilation_growth=token_cnn_dilation_growth,
            token_cnn_dropout=token_cnn_dropout,
            use_stitcher=use_stitcher,
            stitcher_hidden=stitcher_hidden,
            stitcher_kernel=stitcher_kernel,
            stitcher_layers=stitcher_layers,
            stitcher_dropout=stitcher_dropout,
            fusion_type=fusion_type,
            freq_embed_type=freq_embed_type,
            freq_keep_bins=freq_keep_bins,
            theta_dim=theta_dim,
            cond_dim=cond_dim,
            cond_hidden=cond_hidden,
        )
        self.freq_keep_bins = freq_keep_bins
        self.freq_norm = freq_norm
        self.freq_log1p = freq_log1p
        
        # Optional: torch.compile() for additional speedup (PyTorch 2.0+)
        self.use_torch_compile = use_torch_compile
        if use_torch_compile:
            self.gpt = compile_model(self.gpt, mode=compile_mode)

        self.lr = lr
        self.scheduler = scheduler
        self.scheduler_T_0 = scheduler_T_0
        self.scheduler_T_mult = scheduler_T_mult
        self.scheduler_eta_min = scheduler_eta_min
        self.lr_scheduler_interval = lr_scheduler_interval
        self.lr_scheduler_frequency = lr_scheduler_frequency
        self.lr_scheduler_monitor = lr_scheduler_monitor

        # Log-Cosh Loss
        self.log_cosh_loss = LogCoshLoss()

        # ----- Scheduled Sampling state -----
        self.ss_enabled = ss_enabled
        self.ss_warmup_steps = ss_warmup_steps
        self.ss_p_start = ss_p_start
        self.ss_p_end = ss_p_end
        self.ss_unroll_steps = ss_unroll_steps
        self.ss_detach_pred = ss_detach_pred
        self.ss_use_cache = ss_use_cache
        self.ss_focus_fraction = ss_focus_fraction

    def forward(self, x: torch.Tensor, x_fft: torch.Tensor, is_causal: bool = True, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gpt(x, x_fft, is_causal=is_causal, theta=theta)

    # =================================================================
    # Frequency features for scheduled sampling (match rollout + dataset)
    # =================================================================
    def _compute_freq_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Per-token FFT magnitude from time tokens. Uses shared freq_features_from_tokens
        so train / inference / dataset stay in sync (freq_norm, freq_log1p).
        tokens: [B, T, C, K] -> [B, T, C, F]
        """
        return freq_features_from_tokens(
            tokens,
            freq_keep_bins=self.freq_keep_bins,
            freq_log1p=self.freq_log1p,
            freq_norm=self.freq_norm,
        )

    # =================================================================
    # Scheduled sampling probability (linear anneal)
    # =================================================================
    def _ss_probability(self) -> float:
        """Current scheduled-sampling probability (linearly annealed)."""
        step = self.global_step
        if step >= self.ss_warmup_steps:
            return self.ss_p_end
        frac = step / max(self.ss_warmup_steps, 1)
        return self.ss_p_start + (self.ss_p_end - self.ss_p_start) * frac

    # =================================================================
    # Per-token loss helper (shared between full-seq and unroll paths)
    # =================================================================
    def _compute_token_loss(
        self,
        out: torch.Tensor,   # [B, T, K, C] or subset
        y: torch.Tensor,     # same shape
    ) -> torch.Tensor:
        """Compute energy-weighted per-token loss. Input shapes: [B, T, K, C]."""
        eps = 1e-6
        energy = y.pow(2).mean(dim=(2, 3)).sqrt()      # [B, T]
        alpha = 0.5
        weights = (energy + eps) ** alpha
        weights = weights / (weights.mean() + eps)

        residual = out - y
        if self.time_loss == "mse":
            base = residual.pow(2)
        elif self.time_loss == "l1":
            base = residual.abs()
        elif self.time_loss == "log_cosh":
            base = torch.log(torch.cosh(residual + 1e-12))
        else:
            raise ValueError("time_loss must be 'mse', 'l1', or 'log_cosh'")

        base = base.mean(dim=(2, 3))                    # [B, T]
        return (weights * base).mean()

    # =================================================================
    # Main shared step (teacher forcing)
    # =================================================================
    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        x = batch["x"]            # [B, T, C, K]
        x_freq = batch["x_freq"]  # [B, T, ...]
        y = batch["y"]            # [B, T*K, C]
        theta = batch["theta"]    # [B, theta_dim]
        out = self(x, x_freq, is_causal=True, theta=theta)  # [B, T*K, C]

        # ---------------------------------------------------------
        # Reshape to tokens
        # ---------------------------------------------------------
        B, L, C = out.shape
        K = self.hparams.kernel_size
        T = L // K
        if T * K != L:
            raise ValueError("Output length must be multiple of kernel_size")

        out_tok = out.view(B, T, K, C)
        y_tok   = y.view(B, T, K, C)

        # ---------------------------------------------------------
        # Per-token energy (RMS)
        # ---------------------------------------------------------
        eps = 1e-6
        energy = y_tok.pow(2).mean(dim=(2, 3)).sqrt()  # [B, T]

        # ---------------------------------------------------------
        # Soft energy weighting
        # ---------------------------------------------------------
        alpha = 0.5   # <--- IMPORTANT HYPERPARAMETER
        weights = (energy + eps) ** alpha              # [B, T]
        weights = weights / (weights.mean() + eps)     # normalize for stability

        # ---------------------------------------------------------
        # Base per-sample loss (no normalization!)
        # ---------------------------------------------------------
        residual = out_tok - y_tok

        if self.time_loss == "mse":
            base_loss = residual.pow(2)
        elif self.time_loss == "l1":
            base_loss = residual.abs()
        elif self.time_loss == "log_cosh":
            base_loss = torch.log(torch.cosh(residual + 1e-12))
        else:
            raise ValueError("time_loss must be 'mse', 'l1', or 'log_cosh'")

        # base_loss: [B, T, K, C] → reduce over samples
        base_loss = base_loss.mean(dim=(2, 3))         # [B, T]

        # ---------------------------------------------------------
        # Energy-weighted loss
        # ---------------------------------------------------------
        loss = (weights * base_loss).mean()

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        plain_mse = F.mse_loss(out, y)

        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_mse", plain_mse, sync_dist=True)
        self.log(f"{prefix}_energy_mean", energy.mean(), sync_dist=True)
        self.log(f"{prefix}_energy_min", energy.min(), sync_dist=True)
        self.log(f"{prefix}_energy_max", energy.max(), sync_dist=True)

        with torch.no_grad():
            self.log(f"{prefix}_pred_std", out.std(), sync_dist=True)
            self.log(f"{prefix}_target_std", y.std(), sync_dist=True)

        return loss

    # =================================================================
    # Scheduled sampling unroll (training only)
    # =================================================================
    def _scheduled_sampling_step(self, batch: dict) -> torch.Tensor:
        """
        Short autoregressive unroll with scheduled sampling.

        Strategy:
          1. Run full teacher-forced forward to get the base loss (unchanged).
          2. Additionally, do a short unroll of `ss_unroll_steps` tokens from a
             random split point, mixing predictions back with probability p.
          3. Combine the two losses.

        This keeps the teacher-forced gradient signal intact while also exposing
        the model to its own predictions.
        """
        x = batch["x"]            # [B, T, C, K]
        x_freq = batch["x_freq"]  # [B, T, C, F]
        y = batch["y"]            # [B, T*K, C]
        theta = batch["theta"]    # [B, theta_dim]

        B, T_total, C, K = x.shape
        p = self._ss_probability()
        self.log("ss_p", p, prog_bar=True)

        # ---- 1. Full teacher-forced loss (always computed) ----
        tf_loss = self._shared_step(batch, "train")

        # Not enough tokens to unroll — just return teacher-forced loss
        if T_total < self.ss_unroll_steps + 2:
            return tf_loss

        # ---- 2. Short autoregressive unroll ----
        y_tok = y.view(B, T_total, K, C)  # [B, T, K, C]

        # Pick a random starting point (optionally focus on last X% of tokens, e.g. merger)
        max_start = T_total - self.ss_unroll_steps - 1
        start_min = max(0, min(max_start, int(T_total * (1.0 - self.ss_focus_fraction))))
        start = torch.randint(start_min, max_start + 1, (1,)).item()

        # Context: everything up to `start` (teacher-forced embedding)
        # We do a full forward up to `start` to get the transformer hidden states,
        # then unroll from there. For simplicity (and to avoid re-implementing the
        # full embedding pipeline token-by-token), we use the model's forward_step
        # with KV-cache if enabled, or a simple per-step forward otherwise.

        unroll_losses = []
        cond = None
        if self.gpt.theta_dim > 0 and theta is not None:
            cond = self.gpt.cond_mlp(theta)

        if self.ss_use_cache:
            # --- Cache-accelerated unroll ---
            cache = self.gpt.init_kv_cache(B, x.device, x.dtype)

            # Feed context tokens [0, start] into cache
            for t in range(start + 1):
                _, cache = self.gpt.forward_step(
                    x[:, t:t+1], x_freq[:, t:t+1], cache, theta=theta
                )

            # Unroll for ss_unroll_steps tokens
            cur_time = x[:, start:start+1]    # [B, 1, C, K]
            cur_freq = x_freq[:, start:start+1]
            for s in range(self.ss_unroll_steps):
                idx = start + 1 + s
                if idx >= T_total:
                    break

                pred, cache = self.gpt.forward_step(
                    cur_time, cur_freq, cache, theta=theta
                )  # pred: [B, K, C]

                # Loss for this token: pred [B,K,C] → [B,1,K,C]
                gt = y_tok[:, idx]  # [B, K, C]
                token_loss = self._compute_token_loss(
                    pred.unsqueeze(1),  # [B, 1, K, C]
                    gt.unsqueeze(1),
                )
                unroll_losses.append(token_loss)

                # Decide: use prediction or ground truth as next input (match inference rollout)
                if torch.rand(1).item() < p:
                    # Use model's own prediction — freq from same token (match rollout)
                    feedback = pred.detach() if self.ss_detach_pred else pred
                    cur_time = feedback.unsqueeze(1).permute(0, 1, 3, 2)  # [B,1,C,K]
                    cur_freq = self._compute_freq_features(cur_time)  # [B,1,C,F]
                else:
                    # Use ground truth (teacher forcing)
                    if idx < T_total:
                        cur_time = x[:, idx:idx+1]
                        cur_freq = x_freq[:, idx:idx+1]
        else:
            # --- Simple unroll without cache (re-embeds each step) ---
            # Less efficient but simpler; uses full forward over a growing window.
            # For small unroll_steps (2-4), the overhead is minimal.

            for s in range(self.ss_unroll_steps):
                idx = start + 1 + s
                if idx >= T_total:
                    break

                # Build the input sequence up to idx (using potentially mixed tokens)
                if s == 0:
                    # First unroll step: use original context up to idx
                    x_window = x[:, :idx+1]
                    f_window = x_freq[:, :idx+1]
                else:
                    # We already have x_window from previous iteration;
                    # append the next token (either pred or gt)
                    x_window = torch.cat([x_window, next_tok_time], dim=1)
                    f_window = torch.cat([f_window, next_tok_freq], dim=1)

                # Full forward over the window (causal attention handles masking)
                out_window = self.gpt(x_window, f_window, is_causal=True, theta=theta)
                # Extract last token's prediction
                last_pred = out_window[:, -K:]  # [B, K, C]

                # Loss against ground truth
                gt = y_tok[:, idx]  # [B, K, C]
                token_loss = self._compute_token_loss(
                    last_pred.unsqueeze(1),  # [B, 1, K, C]
                    gt.unsqueeze(1),
                )
                unroll_losses.append(token_loss)

                # Decide next input token (match inference rollout)
                if torch.rand(1).item() < p:
                    feedback = last_pred.detach() if self.ss_detach_pred else last_pred
                    next_tok_time = feedback.unsqueeze(1).permute(0, 1, 3, 2)  # [B,1,C,K]
                    next_tok_freq = self._compute_freq_features(next_tok_time)  # [B,1,C,F]
                else:
                    next_tok_time = x[:, idx:idx+1]
                    next_tok_freq = x_freq[:, idx:idx+1]

        # ---- 3. Combine losses ----
        if unroll_losses:
            ss_loss = torch.stack(unroll_losses).mean()
            # Weighted combination: teacher-forced loss dominates, unroll is auxiliary
            total_loss = tf_loss + 0.5 * ss_loss
            self.log("train_ss_loss", ss_loss)
        else:
            total_loss = tf_loss

        return total_loss

    # =================================================================
    # Training / validation steps
    # =================================================================
    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        if self.ss_enabled:
            return self._scheduled_sampling_step(batch)
        return self._shared_step(batch, "train")

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        if not self.scheduler:
            return optimizer

        sched_class = getattr(torch.optim.lr_scheduler, self.scheduler, None)
        if sched_class is None:
            raise ValueError(f"Unknown scheduler: {self.scheduler}. Use a PyTorch name, e.g. ReduceLROnPlateau, CosineAnnealingLR.")

        if self.scheduler == "CosineAnnealingLR":
            kwargs = {"T_max": getattr(self.trainer, "max_epochs", 10)}
        elif self.scheduler == "CosineAnnealingWarmRestarts":
            kwargs = {"T_0": self.scheduler_T_0, "T_mult": self.scheduler_T_mult, "eta_min": self.scheduler_eta_min}
        else:
            kwargs = {}

        scheduler = sched_class(optimizer, **kwargs)
        lr_config = {
            "scheduler": scheduler,
            "interval": self.lr_scheduler_interval,
            "frequency": self.lr_scheduler_frequency,
        }
        if self.lr_scheduler_monitor:
            lr_config["monitor"] = self.lr_scheduler_monitor
        return {"optimizer": optimizer, "lr_scheduler": lr_config}