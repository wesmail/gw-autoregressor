# Generic imports
import math
import numpy as np
from typing import Optional, Tuple, List, Dict, Any

# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# =============================================================================
# Flash Attention Diagnostics
# =============================================================================

def check_flash_attention_available() -> Dict[str, bool]:
    """
    Check which SDPA backends are available.
    Call this at startup to verify Flash Attention is enabled.
    """
    info = {
        "flash_sdp_enabled": torch.backends.cuda.flash_sdp_enabled(),
        "mem_efficient_sdp_enabled": torch.backends.cuda.mem_efficient_sdp_enabled(),
        "math_sdp_enabled": torch.backends.cuda.math_sdp_enabled(),
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        # Flash Attention requires compute capability >= 8.0 (Ampere+)
        major, minor = torch.cuda.get_device_capability(0)
        info["compute_capability"] = f"{major}.{minor}"
        info["supports_flash_attention"] = major >= 8
    return info


def print_flash_attention_status():
    """Pretty print Flash Attention availability."""
    info = check_flash_attention_available()
    print("=" * 60)
    print("SDPA / Flash Attention Status")
    print("=" * 60)
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("=" * 60)
    if info.get("supports_flash_attention") and info.get("flash_sdp_enabled"):
        print("✅ Flash Attention SHOULD be active for bf16/fp16 without explicit mask")
    else:
        print("⚠️  Flash Attention may NOT be available - check GPU & PyTorch version")
    print()

# =============================================================================
# Rotary Position Embedding (RoPE)
# =============================================================================

class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) from the RoFormer paper.
    Applies rotation to query and key vectors based on their position,
    enabling relative position awareness without explicit position embeddings.
    """
    def __init__(self, dim: int, max_len: int = 10000, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.base = base
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._cos_cache = None
        self._sin_cache = None
        self._cache_len = 0

    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        if self._cos_cache is not None and self._cache_len >= seq_len:
            return
        t = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        self._cos_cache = emb.cos().unsqueeze(0).unsqueeze(0).to(dtype)
        self._sin_cache = emb.sin().unsqueeze(0).unsqueeze(0).to(dtype)
        self._cache_len = seq_len

    def forward(self, q: torch.Tensor, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, num_heads, seq_len, head_dim)
        """
        seq_len = q.size(2)
        self._update_cache(seq_len, q.device, q.dtype)
        cos = self._cos_cache[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cache[:, :, :seq_len, :].to(q.dtype)
        return self._apply_rotary(q, cos, sin), self._apply_rotary(k, cos, sin)

    @staticmethod
    def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        cos = cos[..., : x1.size(-1)]
        sin = sin[..., : x1.size(-1)]
        return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# =============================================================================
# Adaptive Normalization
# =============================================================================
class AdaLayerNorm(nn.Module):
    """
    Adaptive LayerNorm (AdaLN / FiLM-LN):
        y = LN(x) * (1 + gamma(cond)) + beta(cond)

    - x:    [B, T, d_model]
    - cond: [B, cond_dim]
    """
    def __init__(self, d_model: int, cond_dim: int):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.to_gamma = nn.Linear(cond_dim, d_model, bias=True)
        self.to_beta  = nn.Linear(cond_dim, d_model, bias=True)

        # Start as identity transform (no conditioning effect at init)
        nn.init.zeros_(self.to_gamma.weight)
        nn.init.zeros_(self.to_gamma.bias)
        nn.init.zeros_(self.to_beta.weight)
        nn.init.zeros_(self.to_beta.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        y = self.ln(x)
        gamma = self.to_gamma(cond).unsqueeze(1)  # [B, 1, d_model]
        beta  = self.to_beta(cond).unsqueeze(1)   # [B, 1, d_model]
        return y * (1.0 + gamma) + beta


# =============================================================================
# Transformer Encoder Layer with RoPE (Flash Attention optimized)
# =============================================================================
class RoPETransformerEncoderLayer(nn.Module):
    """
    Pre-norm transformer encoder layer with RoPE applied to Q, K in attention.
    
    OPTIMIZATION: Uses F.scaled_dot_product_attention with is_causal=True
    and NO explicit mask, allowing Flash Attention to activate.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        cond_dim: int = 0, # condition dimension
    ):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        assert self.head_dim * nhead == d_model and self.head_dim % 2 == 0, \
            "d_model must be divisible by nhead; head_dim must be even for RoPE"

        self.cond_dim = int(cond_dim)
        if self.cond_dim > 0:
            self.self_attn_norm = AdaLayerNorm(d_model, self.cond_dim)
            self.ffn_norm = AdaLayerNorm(d_model, self.cond_dim)
        else:
            self.self_attn_norm = nn.LayerNorm(d_model)
            self.ffn_norm = nn.LayerNorm(d_model)

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.rope = RotaryPositionalEmbedding(dim=self.head_dim)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.attn_dropout_p = dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,
        is_causal: bool = False,
        cond: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            src: Input tensor of shape [B, T, d_model]
            is_causal: If True, applies causal masking via SDPA's is_causal flag
                       (NO explicit mask - this enables Flash Attention)
        """
        # Pre-norm self-attention
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            x = self.self_attn_norm(src, cond)
        else:
            x = self.self_attn_norm(src)

        B, T, _ = x.shape
        q = self.w_q(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)  # [B, H, T, D]
        k = self.w_k(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        v = self.w_v(x).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        q, k = self.rope(q, k)

        # CRITICAL: Do NOT pass attn_mask when is_causal=True
        # This allows Flash Attention to be used
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # ← No explicit mask!
            dropout_p=self.attn_dropout_p if self.training else 0.0,
            is_causal=is_causal,  # ← SDPA handles causal masking internally
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        src = src + self.dropout(self.w_o(attn_out))
        
        # Pre-norm FFN
        if self.cond_dim > 0:
            if cond is None:
                raise ValueError("cond must be provided when cond_dim > 0")
            src = src + self.ffn(self.ffn_norm(src, cond))
        else:
            src = src + self.ffn(self.ffn_norm(src))
        return src


# =============================================================================
# Per-token causal CNN embedding (within-token axis K)
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F

class TokenEmbedding(nn.Module):
    """
    Within-token CNN + attention pooling over K.

    Input:  x [B, T, C, K]
    Output: e [B, T, d_model]

    - No causality over K (full token is observed context).
    - Dilated convs with symmetric padding keep length K unchanged.
    - Attention pooling learns which within-token positions matter.
    """
    def __init__(
        self,
        in_channels: int,
        d_model: int,
        kernel_size: int = 7,
        num_layers: int = 4,
        dilation_growth: int = 2,
        dropout: float = 0.0,
        act: nn.Module = nn.GELU(),
        attn_dropout: float = 0.0,
        use_last: bool = True,
    ):
        super().__init__()
        assert kernel_size >= 2
        assert num_layers >= 1

        self.d_model = d_model
        self.use_last = use_last
        self.kernel_size = int(kernel_size)
        self.dilation_growth = int(dilation_growth)

        self.in_proj = nn.Conv1d(in_channels, d_model, kernel_size=1, bias=False)

        blocks = []
        for i in range(num_layers):
            d = self.dilation_growth ** i  # 1,2,4,8,...
            pad = (d * (self.kernel_size - 1)) // 2  # symmetric "same" padding for odd kernels

            blocks.append(
                nn.ModuleDict(dict(
                    conv=nn.Conv1d(
                        d_model, d_model,
                        kernel_size=self.kernel_size,
                        dilation=d,
                        padding=pad,
                        bias=False,
                    ),
                    norm=nn.GroupNorm(1, d_model),
                    act=act if isinstance(act, nn.Module) else act(),
                    drop=nn.Dropout(dropout),
                    pw=nn.Conv1d(d_model, d_model, kernel_size=1, bias=False),
                ))
            )
        self.blocks = nn.ModuleList(blocks)

        # Attention pooling over K
        self.pool_scorer = nn.Linear(d_model, 1, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)

        # Projection after pooling (optionally concat with last)
        in_dim = (2 * d_model) if use_last else d_model
        self.summary_proj = nn.Linear(in_dim, d_model)
        self.out_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected x as [B, T, C, K], got {x.shape}")

        B, T, C, K = x.shape

        # [B*T, C, K]
        h = x.reshape(B * T, C, K)
        h = self.in_proj(h)  # [B*T, d_model, K]

        # Non-causal residual blocks over within-token time K
        for blk in self.blocks:
            y = blk["conv"](h)   # symmetric padding keeps length
            y = blk["norm"](y)
            y = blk["act"](y)
            y = blk["drop"](y)
            y = blk["pw"](y)
            h = h + y

        # [B*T, d_model, K] -> [B*T, K, d_model]
        h_k = h.transpose(1, 2).contiguous()

        # Attention weights over K
        logits = self.pool_scorer(h_k).squeeze(-1)  # [B*T, K]
        w = torch.softmax(logits, dim=-1)           # [B*T, K]
        w = self.attn_drop(w)

        # Weighted sum: [B*T, d_model]
        tok_att = (h_k * w.unsqueeze(-1)).sum(dim=1)

        if self.use_last:
            tok_last = h[..., -1]  # [B*T, d_model]
            tok_cat = torch.cat([tok_att, tok_last], dim=-1)
        else:
            tok_cat = tok_att

        tok = self.summary_proj(tok_cat)
        tok = self.out_norm(tok)

        return tok.view(B, T, self.d_model)


# =============================================================================
# Causal Stitcher 1D
# =============================================================================

class CausalStitcher1D(nn.Module):
    """
    Small causal Conv1D residual stack operating on sample axis.
    Input/Output: [B, C, L]
    """
    def __init__(self, channels: int, hidden: int = 64, kernel_size: int = 9, num_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        assert kernel_size >= 2
        self.kernel_size = int(kernel_size)

        self.in_proj = nn.Conv1d(channels, hidden, kernel_size=1, bias=False)
        self.blocks = nn.ModuleList()
        for _ in range(num_layers):
            self.blocks.append(nn.Sequential(
                nn.Conv1d(hidden, hidden, kernel_size=self.kernel_size, padding=0, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Conv1d(hidden, hidden, kernel_size=1, bias=False),
            ))
        self.out_proj = nn.Conv1d(hidden, channels, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, L]
        h = self.in_proj(x)
        for blk in self.blocks:
            h_pad = F.pad(h, (self.kernel_size - 1, 0))  # left pad only (causal)
            h = h + blk(h_pad)
        y = self.out_proj(h)
        return y


# =============================================================================
# GPT Model (Flash Attention Optimized)
# =============================================================================

class GPT(nn.Module):
    """
    Uni-modal GPT model (time-domain only).
    Uses TokenEmbedding over K (within-token time). No frequency-domain branch.
    
    OPTIMIZATIONS:
    - Flash Attention enabled via is_causal flag (no explicit mask)
    - Optional torch.compile() support
    """

    def __init__(
        self,
        in_channels: int = 3,
        kernel_size: int = 16,
        d_model: int = 128,
        num_heads: int = 2,
        num_enc_layers: int = 2,
        dropout: float = 0.1,
        max_len: int = 5000,
        dim_feedforward_multiplier: int = 4,
        # Token CNN embed config
        token_cnn_kernel: int = 7,
        token_cnn_layers: int = 4,
        token_cnn_dilation_growth: int = 2,
        token_cnn_dropout: float = 0.0,
        # Post-head stitcher
        use_stitcher: bool = True,
        stitcher_hidden: int = 64,
        stitcher_kernel: int = 9,
        stitcher_layers: int = 4,
        stitcher_dropout: float = 0.0,
        # conditioning
        theta_dim: int = 0,
        cond_dim: int = 128,
        cond_hidden: int = 256,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.kernel_size = kernel_size

        self.d_model = d_model
        self.num_heads = num_heads
        self.num_enc_layers = num_enc_layers
        self.dropout = dropout
        self.max_len = max_len
        self.dim_feedforward = self.d_model * dim_feedforward_multiplier

        # Post-head stitcher config
        self.use_stitcher = use_stitcher
        self.stitcher_hidden = stitcher_hidden
        self.stitcher_kernel = stitcher_kernel
        self.stitcher_layers = stitcher_layers
        self.stitcher_dropout = stitcher_dropout

        self.theta_dim = int(theta_dim)
        self.cond_dim = int(cond_dim)

        if self.theta_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.theta_dim, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, self.cond_dim),
            )
        else:
            self.cond_mlp = None

        # Time-domain token embedding only (uni-modal)
        self.time_token_embed = TokenEmbedding(
            in_channels=in_channels,
            d_model=self.d_model,
            kernel_size=token_cnn_kernel,
            num_layers=token_cnn_layers,
            dilation_growth=token_cnn_dilation_growth,
            dropout=token_cnn_dropout,
            act=nn.GELU(),
        )

        # Stack of RoPE encoder layers (Flash Attention optimized)
        layer_cond_dim = self.cond_dim if self.theta_dim > 0 else 0
        self.encoder_layers = nn.ModuleList([
            RoPETransformerEncoderLayer(
                d_model=self.d_model,
                nhead=num_heads,
                dim_feedforward=self.dim_feedforward,
                dropout=dropout,
                cond_dim=layer_cond_dim,
            )
            for _ in range(num_enc_layers)
        ])

        self.pred_head = nn.Linear(self.d_model, in_channels * kernel_size)
        self.dropout_layer = nn.Dropout(p=dropout)

        if self.use_stitcher:
            self.stitcher = CausalStitcher1D(
                channels=self.in_channels,
                hidden=stitcher_hidden,
                kernel_size=stitcher_kernel,
                num_layers=stitcher_layers,
                dropout=stitcher_dropout,
            )

    def forward(self, x_time: torch.Tensor,
                x_freq: Optional[torch.Tensor] = None,
                is_causal: bool = False,
                theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_time: Input tensor of shape [B, T, C, K]
            x_freq: Unused (kept for API compatibility with eval script; uni-modal uses time only).
            is_causal: If True, uses causal attention (for autoregressive training)
            theta: Optional conditioning [B, theta_dim]
        
        Returns:
            Predictions of shape [B, T*K, C]
        """
        if x_time.dim() != 4:
            raise ValueError(f"Expected 4D input [B, T, C, K], got {x_time.shape}")

        B, T, C, K = x_time.shape
        if C != self.in_channels or K != self.kernel_size:
            raise ValueError(
                f"Input has C={C},K={K} but model expects C={self.in_channels},K={self.kernel_size}"
            )

        # Time-domain token embeddings only (uni-modal): [B, T, d_model]
        h = self.time_token_embed(x_time)
        h = self.dropout_layer(h)

        cond = None
        if self.theta_dim > 0:
            if theta is None:
                raise ValueError("theta must be provided when theta_dim > 0")
            if theta.dim() != 2 or theta.size(-1) != self.theta_dim:
                raise ValueError(f"theta must be [B, {self.theta_dim}], got {theta.shape}")
            cond = self.cond_mlp(theta)  # [B, cond_dim]


        # OPTIMIZED: No explicit causal mask - just pass is_causal flag
        # This allows Flash Attention to be used when available
        for layer in self.encoder_layers:
            h = layer(h, is_causal=is_causal, cond=cond)

        # Single output (mean prediction): [B, T, C*K] -> [B, T*K, C]
        out = self.pred_head(h)                      # [B, T, C*K]
        out = out.view(B, T, K, C).contiguous()      # [B, T, K, C]
        out = out.view(B, T * K, C)                  # [B, T*K, C]

        if self.use_stitcher:
            out = self.stitcher(out.transpose(1, 2)).transpose(1, 2)  # [B, C, L] -> [B, L, C]

        return out

# =============================================================================
# Utility: Compile model with torch.compile() for extra speedup
# =============================================================================

def compile_model(model: nn.Module, mode: str = "reduce-overhead") -> nn.Module:
    """
    Wrap model with torch.compile() for additional speedup.
    
    Args:
        model: The model to compile
        mode: Compilation mode. Options:
            - "default": Good balance of compile time and speedup
            - "reduce-overhead": Best for small batches / inference
            - "max-autotune": Slower compile, potentially faster runtime
    
    Returns:
        Compiled model (or original if torch.compile unavailable)
    """
    if hasattr(torch, "compile"):
        print(f"Compiling model with mode='{mode}'...")
        return torch.compile(model, mode=mode)
    else:
        print("torch.compile() not available (requires PyTorch 2.0+)")
        return model