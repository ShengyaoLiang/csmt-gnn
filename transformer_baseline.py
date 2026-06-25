"""Pure causal Transformer baseline for CSMT-GNN diagnostics.

This module is intentionally separate from csmt_gnn.py.  The point is to keep a
plain token-only Transformer baseline that does not inherit CSMT block pooling,
AST gates, graph injection, or CVD paths through configuration switches.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from csmt_gnn import ExpertMLP, RMSNorm, _is_real_number, _validate_id_tensor, _validate_length_tensor


@dataclass
class TransformerBaselineConfig:
    vocab_size: int = 50000
    num_layers: int = 1
    hidden_size: int = 768
    max_tokens: int = 2048
    num_heads: int = 12
    ffn_multiplier: float = 2.0
    dropout: float = 0.0
    validate_input_ranges: bool = True
    tie_embeddings: bool = False

    def __post_init__(self) -> None:
        for name in ("vocab_size", "num_layers", "hidden_size", "max_tokens", "num_heads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        for name in ("ffn_multiplier", "dropout"):
            value = getattr(self, name)
            if not _is_real_number(value):
                raise TypeError(f"{name} must be a finite real number, got {value!r}")
        for name in ("validate_input_ranges", "tie_embeddings"):
            value = getattr(self, name)
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if int(self.hidden_size * self.ffn_multiplier) <= 0:
            raise ValueError("hidden_size * ffn_multiplier must produce at least one FFN channel")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")


class CausalSelfAttention(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.qkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, length, hidden = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        causal = torch.ones(length, length, dtype=torch.bool, device=x.device).tril()
        key_valid = valid_mask.view(batch, 1, 1, length)
        attn_mask = causal.view(1, 1, length, length) & key_valid
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )
        y = y.transpose(1, 2).contiguous().view(batch, length, hidden)
        return self.out_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        hidden = config.hidden_size
        ffn_hidden = int(hidden * config.ffn_multiplier)
        self.norm_attn = RMSNorm(hidden)
        self.attn = CausalSelfAttention(config)
        self.norm_ffn = RMSNorm(hidden)
        self.ffn = ExpertMLP(hidden, ffn_hidden)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm_attn(x), valid_mask)
        x = x + self.ffn(self.norm_ffn(x))
        return x.masked_fill(~valid_mask[..., None], 0)


class TinyCausalTransformer(nn.Module):
    """A standard token-only causal Transformer used as a diagnostic baseline."""

    def __init__(self, config: TransformerBaselineConfig) -> None:
        super().__init__()
        self.config = config
        hidden = config.hidden_size
        self.token_embed = nn.Embedding(config.vocab_size, hidden)
        self.pos_embed = nn.Embedding(config.max_tokens, hidden)
        self.layers = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(hidden)
        self.output_head = nn.Linear(hidden, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output_head.weight = self.token_embed.weight

    def forward(self, input_ids: torch.Tensor, lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        squeeze_output = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze_output = True
        elif input_ids.dim() != 2:
            raise ValueError("TinyCausalTransformer expects input_ids shaped [L] or [batch, L].")

        input_ids = input_ids[:, : self.config.max_tokens]
        _validate_id_tensor("input_ids", input_ids, self.config.vocab_size, self.config.validate_input_ranges)
        input_ids = input_ids.long()
        batch, length = input_ids.shape
        if length == 0:
            raise ValueError("input_ids must contain at least one token after max_tokens truncation.")
        if lengths is None:
            lengths = torch.full((batch,), length, dtype=torch.long, device=input_ids.device)
        else:
            _validate_length_tensor("lengths", lengths)
            lengths = lengths.to(device=input_ids.device).view(-1).long()
            if lengths.numel() != batch:
                raise ValueError(f"lengths must have batch dimension {batch}, got {lengths.numel()}.")
            if bool((lengths < 0).any()) or bool((lengths > length).any()):
                min_len = int(lengths.min().item())
                max_len = int(lengths.max().item())
                raise ValueError(f"lengths must be in [0, {length}] after max_tokens truncation, got min={min_len}, max={max_len}.")

        positions = torch.arange(length, device=input_ids.device).clamp_max(self.config.max_tokens - 1)
        valid_mask = positions.view(1, -1) < lengths.view(-1, 1)
        x = self.token_embed(input_ids) + self.pos_embed(positions).unsqueeze(0)
        x = x.masked_fill(~valid_mask[..., None], 0)
        for layer in self.layers:
            x = layer(x, valid_mask)
        logits = self.output_head(self.final_norm(x))
        return logits.squeeze(0) if squeeze_output else logits
