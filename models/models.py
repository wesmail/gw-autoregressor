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

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        offset: int = 0,  # NEW: position offset for KV-cache decoding
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        q, k: (B, num_heads, seq_len, head_dim)
        offset: starting position index (used during cached inference so that
                 a single new token at step t gets the correct positional encoding).
        """
        seq_len = q.size(2)
        total_len = offset + seq_len
        self._update_cache(total_len, q.device, q.dtype)
        cos = self._cos_cache[:, :, offset:total_len, :].to(q.dtype)
        sin = self._sin_cache[:, :, offset:total_len, :].to(q.dtype)
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
        Full-sequence forward (unchanged from original).
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

    # -----------------------------------------------------------------
    # KV-CACHE: single-step forward for incremental decoding
    # -----------------------------------------------------------------
    def forward_cached(
        self,
        src: torch.Tensor,               # [B, 1, d_model]  (single new token)
        cache: Dict[str, torch.Tensor],   # {"k": [B,H,T_past,D], "v": ...}
        offset: int,                      # number of tokens already in cache
        cond: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Incremental (cached) forward for one new token.

        Args:
            src:    [B, 1, d_model] — the new token's hidden state.
            cache:  per-layer dict with "k" and "v" tensors from previous steps.
            offset: how many tokens are already cached (= time index of new token).
            cond:   [B, cond_dim] conditioning vector (or None).

        Returns:
            (output, updated_cache)
        """
        # Pre-norm
        if self.cond_dim > 0:
            x = self.self_attn_norm(src, cond)
        else:
            x = self.self_attn_norm(src)

        B = x.size(0)
        # Q, K, V for the single new token  [B, H, 1, D]
        q = self.w_q(x).view(B, 1, self.nhead, self.head_dim).transpose(1, 2)
        k_new = self.w_k(x).view(B, 1, self.nhead, self.head_dim).transpose(1, 2)
        v_new = self.w_v(x).view(B, 1, self.nhead, self.head_dim).transpose(1, 2)

        # Apply RoPE with correct position offset
        q, k_new = self.rope(q, k_new, offset=offset)

        # Concatenate new K/V to cache
        if cache["k"].size(2) == 0:
            k_all = k_new
            v_all = v_new
        else:
            k_all = torch.cat([cache["k"], k_new], dim=2)  # [B,H,T_past+1,D]
            v_all = torch.cat([cache["v"], v_new], dim=2)

        # Update cache
        cache = {"k": k_all, "v": v_all}

        # Attention: Q(new) vs K/V(all past + new) — no causal mask needed
        # because new token is the last one and cache only has past tokens.
        attn_out = F.scaled_dot_product_attention(
            q, k_all, v_all,
            attn_mask=None,
            dropout_p=0.0,       # no dropout at inference
            is_causal=False,     # single query against all past — no future exists
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, 1, self.d_model)
        src = src + self.w_o(attn_out)

        # Pre-norm FFN
        if self.cond_dim > 0:
            src = src + self.ffn(self.ffn_norm(src, cond))
        else:
            src = src + self.ffn(self.ffn_norm(src))
        return src, cache


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
# Lightweight MLP embedding for frequency features (small F bins)
# =============================================================================

class FreqMLPEmbed(nn.Module):
    """
    Maps per-token frequency features [B, T, C, F] -> [B, T, d_model].

    TokenEmbedding (conv-based) is over-parameterized for small F (e.g. F=8).
    A simple Linear+GELU+LayerNorm is cheaper and sufficient for low-dimensional
    spectral summaries.
    """

    def __init__(self, in_channels: int, freq_bins: int, d_model: int):
        super().__init__()
        in_dim = in_channels * freq_bins  # flatten C*F per token
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
        )

    def forward(self, x_freq: torch.Tensor) -> torch.Tensor:
        """x_freq: [B, T, C, F] -> [B, T, d_model]"""
        B, T, C, F = x_freq.shape
        return self.net(x_freq.reshape(B, T, C * F))


# =============================================================================
# GPT Model (Flash Attention Optimized, KV-Cache for Inference)
# =============================================================================

class GPT(nn.Module):
    """
    GPT model for seismic tokens.
    Uses TokenCausalEmbedding over K (within-token time) instead of 1x1 conv "Embedding".
    
    OPTIMIZATIONS:
    - Flash Attention enabled via is_causal flag (no explicit mask)
    - Optional torch.compile() support
    - KV-cache for fast autoregressive rollout inference
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

        # multi-modal fusion type
        fusion_type: str = "cross_attention",
        # frequency branch: "mlp" (lightweight) or "conv" (legacy TokenEmbedding)
        freq_embed_type: str = "mlp",
        freq_keep_bins: int = 8,
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

        # Multi-modal fusion type
        self.fusion_type = fusion_type

        self.theta_dim = int(theta_dim)
        self.cond_dim = int(cond_dim)
        self.freq_keep_bins = int(freq_keep_bins)

        if self.theta_dim > 0:
            self.cond_mlp = nn.Sequential(
                nn.Linear(self.theta_dim, cond_hidden),
                nn.SiLU(),
                nn.Linear(cond_hidden, self.cond_dim),
            )
        else:
            self.cond_mlp = None

        # Time-domain token embedding (causal CNN over K)
        self.time_token_embed = TokenEmbedding(
            in_channels=in_channels,
            d_model=self.d_model,
            kernel_size=token_cnn_kernel,
            num_layers=token_cnn_layers,
            dilation_growth=token_cnn_dilation_growth,
            dropout=token_cnn_dropout,
            act=nn.GELU(),
        )

        # Frequency embedding: MLP (lightweight for small F) or conv (legacy)
        if freq_embed_type == "mlp":
            self.frequency_token_embed = FreqMLPEmbed(
                in_channels=in_channels,
                freq_bins=freq_keep_bins,
                d_model=self.d_model,
            )
        elif freq_embed_type == "conv":
            self.frequency_token_embed = TokenEmbedding(
                in_channels=in_channels,
                d_model=self.d_model,
                kernel_size=token_cnn_kernel,
                num_layers=token_cnn_layers,
                dilation_growth=token_cnn_dilation_growth,
                dropout=token_cnn_dropout,
                act=nn.GELU(),
            )
        else:
            raise ValueError(f"Unknown freq_embed_type='{freq_embed_type}'")

        # Fusion layer
        if fusion_type == "concat":
            self.fusion = nn.Linear(self.d_model * 2, self.d_model)
        elif fusion_type == "add":
            self.fusion = nn.LayerNorm(self.d_model)
        elif fusion_type == "cross_attention":
            self.cross_attn = nn.MultiheadAttention(
                embed_dim=self.d_model,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.fusion = nn.LayerNorm(self.d_model)
        else:
            raise ValueError(f"Unknown fusion_type='{fusion_type}'.")           

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

    # =================================================================
    # ORIGINAL: Full-sequence forward (training / teacher-forced eval)
    # =================================================================
    def forward(self, x_time: torch.Tensor, 
                x_freq: torch.Tensor, 
                is_causal: bool = False, 
                theta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x_time: Input tensor of shape [B, T, C, K]
            x_freq: Input tensor of shape [B, T, C, F]
            is_causal: If True, uses causal attention (for autoregressive training)
        
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

        # Time-domain token embeddings: [B, T, d_model]
        h_time = self.time_token_embed(x_time)
        h_time = self.dropout_layer(h_time)

        # Frequency-domain token embeddings: [B, T, d_model]
        h_freq = self.frequency_token_embed(x_freq)
        h_freq = self.dropout_layer(h_freq)

        # Fuse modalities
        if self.fusion_type == "concat":
            h = torch.cat([h_time, h_freq], dim=-1)  # [B, T, d_model*2]
            h = self.fusion(h)  # [B, T, d_model]
        elif self.fusion_type == "add":
            h = self.fusion(h_time + h_freq)  # [B, T, d_model]
        elif self.fusion_type == "cross_attention":
            # Time attends to frequency
            T = h_time.size(1)
            attn_mask = torch.triu(torch.ones(T, T, device=h_time.device, dtype=torch.bool), diagonal=1)
            h_cross, _ = self.cross_attn(h_time, h_freq, h_freq, attn_mask=attn_mask)
            h = self.fusion(h_time + h_cross)  # [B, T, d_model]

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

    # =================================================================
    # KV-CACHE: Helper types and methods for fast autoregressive rollout
    # =================================================================

    # Cache type: list of per-layer dicts, one per encoder layer.
    # Each dict has {"k": Tensor[B,H,T_cached,D], "v": Tensor[B,H,T_cached,D]}.
    KVCache = List[Dict[str, torch.Tensor]]

    def init_kv_cache(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> "GPT.KVCache":
        """
        Create an empty KV cache (one entry per encoder layer).
        """
        cache: GPT.KVCache = []
        for _ in self.encoder_layers:
            cache.append({
                "k": torch.zeros(batch_size, self.num_heads, 0, self.d_model // self.num_heads,
                                 device=device, dtype=dtype),
                "v": torch.zeros(batch_size, self.num_heads, 0, self.d_model // self.num_heads,
                                 device=device, dtype=dtype),
            })
        return cache

    def _embed_single_token(
        self,
        x_time: torch.Tensor,   # [B, 1, C, K]
        x_freq: torch.Tensor,   # [B, 1, C, F]
    ) -> torch.Tensor:
        """
        Embed a single token pair (time + freq) and fuse.
        Returns: [B, 1, d_model]
        """
        h_time = self.time_token_embed(x_time)       # [B, 1, d_model]
        h_freq = self.frequency_token_embed(x_freq)   # [B, 1, d_model]

        # Fuse — for single token, cross-attention degenerates to add
        if self.fusion_type == "concat":
            h = torch.cat([h_time, h_freq], dim=-1)
            h = self.fusion(h)
        elif self.fusion_type == "add":
            h = self.fusion(h_time + h_freq)
        elif self.fusion_type == "cross_attention":
            # Single-token: cross-attn with T=1 is just element-wise, so fallback to add
            h = self.fusion(h_time + h_freq)
        return h

    def forward_step(
        self,
        x_time_new: torch.Tensor,     # [B, 1, C, K] single new time token
        x_freq_new: torch.Tensor,     # [B, 1, C, F] single new freq token
        cache: "GPT.KVCache",
        theta: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, "GPT.KVCache"]:
        """
        Incremental forward for ONE new token using KV cache.
        Used during autoregressive rollout inference.

        Args:
            x_time_new: [B, 1, C, K]
            x_freq_new: [B, 1, C, F]
            cache:      list of per-layer {"k", "v"} from previous steps.
            theta:      [B, theta_dim] conditioning (optional).

        Returns:
            y_new:  [B, K, C]  — predicted samples for this token.
            cache:  updated cache (keys/values appended for new token).
        """
        # 1. Embed the new token  [B, 1, d_model]
        h = self._embed_single_token(x_time_new, x_freq_new)

        # 2. Conditioning
        cond = None
        if self.theta_dim > 0:
            if theta is None:
                raise ValueError("theta must be provided when theta_dim > 0")
            cond = self.cond_mlp(theta)  # [B, cond_dim]

        # 3. Run through each layer with cache
        offset = cache[0]["k"].size(2)  # number of tokens already cached
        new_cache: GPT.KVCache = []
        for layer, layer_cache in zip(self.encoder_layers, cache):
            h, updated = layer.forward_cached(h, layer_cache, offset=offset, cond=cond)
            new_cache.append(updated)

        # 4. Prediction head  [B, 1, C*K] → [B, K, C]
        out = self.pred_head(h)  # [B, 1, C*K]
        B = out.size(0)
        out = out.view(B, self.kernel_size, self.in_channels)  # [B, K, C]

        # NOTE: stitcher is NOT applied per-step; it operates on the full
        # output sequence and should be applied after rollout if needed.
        return out, new_cache

    @torch.no_grad()
    def rollout_cached(
        self,
        context_time: torch.Tensor,     # [B, T_ctx, C, K]
        context_freq: torch.Tensor,     # [B, T_ctx, C, F]
        n_future: int,
        theta: Optional[torch.Tensor] = None,
        freq_from_pred: Optional[Any] = None,  # optional callable: pred_samples → freq token
    ) -> torch.Tensor:
        """
        Autoregressive rollout using KV cache.

        1. Feeds context tokens one-by-one to populate the cache.
        2. Generates `n_future` tokens step-by-step.

        Args:
            context_time: [B, T_ctx, C, K] context tokens (time domain).
            context_freq: [B, T_ctx, C, F] context tokens (freq domain).
            n_future:     number of future tokens to generate.
            theta:        [B, theta_dim] conditioning.
            freq_from_pred: optional callable(pred [B,K,C]) → freq_tok [B,1,C,F].
                            If None, uses zeros for freq during generation.

        Returns:
            preds: [B, n_future, K, C] — predicted future tokens (before stitcher).
        """
        B, T_ctx, C, K = context_time.shape
        device = context_time.device
        dtype = context_time.dtype

        cache = self.init_kv_cache(B, device, dtype)

        # --- Phase 1: feed context tokens to build the cache ---
        for t in range(T_ctx):
            xt = context_time[:, t : t + 1]  # [B, 1, C, K]
            xf = context_freq[:, t : t + 1]  # [B, 1, C, F]
            _, cache = self.forward_step(xt, xf, cache, theta=theta)

        # --- Phase 2: generate future tokens autoregressively ---
        preds = []
        # Use the last context token's prediction as the first "input" for generation
        # (or re-predict it — here we just start generating from scratch after context)
        # We need an input token for each step; use last context token as seed.
        last_time = context_time[:, -1:]  # [B, 1, C, K]
        last_freq = context_freq[:, -1:]  # [B, 1, C, F]

        for step in range(n_future):
            if step == 0:
                xt = last_time
                xf = last_freq
            else:
                # Build input token from previous prediction
                xt = prev_pred.unsqueeze(1)  # [B, 1, K, C] → need [B, 1, C, K]
                xt = xt.permute(0, 1, 3, 2)  # [B, 1, C, K]
                if freq_from_pred is not None:
                    xf = freq_from_pred(prev_pred)  # user-supplied
                else:
                    F = self.freq_keep_bins
                    xf = torch.zeros(B, 1, C, F, device=device, dtype=dtype)

            pred, cache = self.forward_step(xt, xf, cache, theta=theta)
            preds.append(pred)              # [B, K, C]
            prev_pred = pred                # for next step's input

        # Stack: [B, n_future, K, C]
        return torch.stack(preds, dim=1)

    def apply_stitcher(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Apply stitcher to a batch of predicted tokens (post-rollout utility).
        Args:
            tokens: [B, T, K, C]
        Returns:
            smoothed: [B, T*K, C]
        """
        B, T, K, C = tokens.shape
        out = tokens.reshape(B, T * K, C)
        if self.use_stitcher:
            out = self.stitcher(out.transpose(1, 2)).transpose(1, 2)
        return out


# =============================================================================
# KV-Cache Sanity Check
# =============================================================================

@torch.no_grad()
def sanity_check_kv_cache(model: GPT, device: torch.device = torch.device("cpu")):
    """
    Compare cached step-by-step output vs full-sequence forward.
    Asserts max absolute error < 1e-4 (fp32). Call in eval mode.

    Usage:
        model.eval()
        sanity_check_kv_cache(model, device=torch.device("cuda"))
    """
    model.eval()
    B, T, C, K = 2, 8, model.in_channels, model.kernel_size
    F = model.freq_keep_bins
    dtype = torch.float32

    x_time = torch.randn(B, T, C, K, device=device, dtype=dtype)
    x_freq = torch.randn(B, T, C, F, device=device, dtype=dtype)
    theta = torch.randn(B, model.theta_dim, device=device, dtype=dtype) if model.theta_dim > 0 else None

    # --- Full-sequence forward (no stitcher for fair comparison) ---
    # Temporarily disable stitcher
    orig_stitcher = model.use_stitcher
    model.use_stitcher = False
    full_out = model(x_time, x_freq, is_causal=True, theta=theta)  # [B, T*K, C]
    full_tokens = full_out.view(B, T, K, C)
    model.use_stitcher = orig_stitcher

    # --- Step-by-step cached forward ---
    cache = model.init_kv_cache(B, device, dtype)
    step_preds = []
    for t in range(T):
        xt = x_time[:, t:t+1]
        xf = x_freq[:, t:t+1]
        pred, cache = model.forward_step(xt, xf, cache, theta=theta)
        step_preds.append(pred)  # [B, K, C]
    cached_tokens = torch.stack(step_preds, dim=1)  # [B, T, K, C]

    # --- Compare ---
    max_err = (full_tokens - cached_tokens).abs().max().item()
    print(f"KV-cache sanity check: max |error| = {max_err:.2e}")
    assert max_err < 1e-4, f"KV-cache mismatch! max error = {max_err:.2e}"
    print("✅ KV-cache sanity check passed!")


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