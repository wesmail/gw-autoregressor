"""
PyTorch Lightning module for training the GPT model (time-domain only) with
time-domain loss (MSE/L1) and multi-resolution STFT loss.

OPTIMIZATIONS:
- Flash Attention enabled (via is_causal flag in model)
- Optional torch.compile() for additional speedup
- Flash Attention status logging at init
"""
from typing import Any, Dict, Optional

import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .models import GPT, print_flash_attention_status, compile_model

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
            theta_dim=theta_dim,
            cond_dim=cond_dim,
            cond_hidden=cond_hidden,
        )
        
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

    def forward(self, x: torch.Tensor, x_fft: torch.Tensor, is_causal: bool = True, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gpt(x, x_fft, is_causal=is_causal, theta=theta)

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

        self.log(f"{prefix}_loss", loss, prog_bar=True)
        self.log(f"{prefix}_mse", plain_mse)
        self.log(f"{prefix}_energy_mean", energy.mean())
        self.log(f"{prefix}_energy_min", energy.min())
        self.log(f"{prefix}_energy_max", energy.max())

        with torch.no_grad():
            self.log(f"{prefix}_pred_std", out.std())
            self.log(f"{prefix}_target_std", y.std())

        return loss

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
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