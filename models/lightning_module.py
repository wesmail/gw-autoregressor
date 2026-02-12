"""
PyTorch Lightning module for training the GPT model (time-domain only) with
time-domain loss (MSE/L1/log_cosh) and multi-resolution STFT loss.

OPTIMIZATIONS:
- Flash Attention enabled (via is_causal flag in model)
- Optional torch.compile() for additional speedup
- Flash Attention status logging at init
"""
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning.pytorch import LightningModule

from .models import GPT, print_flash_attention_status, compile_model

# =============================================================================
# Log-Cosh Loss
# =============================================================================
class LogCoshLoss(nn.Module):
    def forward(self, x, y):  # x, y: [B, T, F]
        diff = x - y
        return torch.mean(torch.log(torch.cosh(diff + 1e-12)))


# =============================================================================
# Multi-Resolution STFT Loss
# =============================================================================
class MultiResSTFTLoss(nn.Module):
    """
    Multi-resolution STFT magnitude L1 loss.
    Inputs x, y: [B, C, T]
    """
    def __init__(self, n_ffts: Tuple[int, ...] = (256, 1024, 4096), eps: float = 1e-8):
        super().__init__()
        self.n_ffts = tuple(n_ffts)
        self.eps = eps

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        x, y: [B, C, T]
        """
        B, C, T = x.shape
        loss = x.new_zeros(())

        for n_fft in self.n_ffts:
            hop = n_fft // 4
            win = torch.hann_window(n_fft, device=x.device, dtype=torch.float32)

            x_ = x.reshape(B * C, T)
            y_ = y.reshape(B * C, T)

            X = torch.stft(x_, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=win, center=True, return_complex=True)
            Y = torch.stft(y_, n_fft=n_fft, hop_length=hop, win_length=n_fft,
                           window=win, center=True, return_complex=True)

            magX = (X.abs() + self.eps)
            magY = (Y.abs() + self.eps)

            loss = loss + (magX - magY).abs().mean()

        return (loss / float(len(self.n_ffts))).to(x.dtype)
    

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
        # Loss: weighted sum of time loss + multi-resolution STFT loss
        time_loss: str = "l1",
        time_loss_weight: float = 1.0,
        stft_loss_weight: float = 0.0,
        stft_n_ffts: Union[Tuple[int, ...], list] = (256, 1024, 4096),
        stft_eps: float = 1e-8,
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

        # Log-Cosh Loss (used when time_loss == "log_cosh")
        self.log_cosh_loss = LogCoshLoss()

        # Loss weights
        self.time_loss_weight = float(time_loss_weight)
        self.stft_loss_weight = float(stft_loss_weight)
        n_ffts = tuple(stft_n_ffts) if isinstance(stft_n_ffts, list) else stft_n_ffts
        self.stft_loss_fn = MultiResSTFTLoss(n_ffts=n_ffts, eps=stft_eps)

    def forward(self, x: torch.Tensor, x_freq: Optional[torch.Tensor] = None, is_causal: bool = True, theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.gpt(x, x_freq=x_freq, is_causal=is_causal, theta=theta)

    def _shared_step(self, batch: dict, prefix: str) -> torch.Tensor:
        x = batch["x"]            # [B, T, C, K]
        y = batch["y"]            # [B, T*K, C]
        theta = batch["theta"]    # [B, theta_dim]
        out = self(x, x_freq=None, is_causal=True, theta=theta)  # [B, T*K, C]

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
        # Time loss (energy-weighted)
        # ---------------------------------------------------------
        time_loss_term = (weights * base_loss).mean()

        # ---------------------------------------------------------
        # Multi-resolution STFT loss (inputs [B, C, T])
        # ---------------------------------------------------------
        if self.stft_loss_weight != 0:
            out_bct = out.transpose(1, 2)   # [B, L, C] -> [B, C, L]
            y_bct = y.transpose(1, 2)       # [B, L, C] -> [B, C, L]
            stft_loss_term = self.stft_loss_fn(out_bct, y_bct)
        else:
            stft_loss_term = out.new_zeros(())

        # ---------------------------------------------------------
        # Combined loss
        # ---------------------------------------------------------
        loss = self.time_loss_weight * time_loss_term + self.stft_loss_weight * stft_loss_term

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        plain_mse = F.mse_loss(out, y)
        self.log(f"{prefix}_loss", loss, prog_bar=True, sync_dist=True)
        self.log(f"{prefix}_time_loss", time_loss_term, sync_dist=True)
        self.log(f"{prefix}_mse", plain_mse, sync_dist=True)
        if self.stft_loss_weight != 0:
            self.log(f"{prefix}_stft_loss", stft_loss_term, sync_dist=True)
        self.log(f"{prefix}_energy_mean", energy.mean(), sync_dist=True)
        self.log(f"{prefix}_energy_min", energy.min(), sync_dist=True)
        self.log(f"{prefix}_energy_max", energy.max(), sync_dist=True)

        with torch.no_grad():
            self.log(f"{prefix}_pred_std", out.std(), sync_dist=True)
            self.log(f"{prefix}_target_std", y.std(), sync_dist=True)

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