"""
CSMT-GNN model components.

This file replaces the earlier mixed prototype with a maintainable path:
PyTorch scaled-dot-product attention for block-local Transformer work, compact
offline AST ids with shared trainable embeddings, prefix-masked block graph
attention, and a sorted top-k MoE fallback that avoids per-token dynamic
dispatch.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


_ID_DTYPES = {torch.int32, torch.int64}
_MASK_DTYPES = {torch.bool, torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64}


def _is_real_number(value: object) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _validate_id_tensor(name: str, tensor: torch.Tensor, upper_bound: int, check_range: bool = True) -> None:
    if tensor.dtype not in _ID_DTYPES:
        raise TypeError(f"{name} must contain int32 or int64 ids, got dtype={tensor.dtype}.")
    if tensor.numel() == 0 or not check_range:
        return
    if bool((tensor < 0).any()) or bool((tensor >= upper_bound).any()):
        min_id = int(tensor.min().item())
        max_id = int(tensor.max().item())
        raise ValueError(f"{name} ids must be in [0, {upper_bound - 1}], got min={min_id}, max={max_id}.")


def _validate_mask_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in _MASK_DTYPES:
        raise TypeError(f"{name} must be a bool or integer mask, got dtype={tensor.dtype}.")


def _validate_length_tensor(name: str, tensor: torch.Tensor) -> None:
    if tensor.dtype not in _ID_DTYPES:
        raise TypeError(f"{name} must contain int32 or int64 lengths, got dtype={tensor.dtype}.")


@dataclass
class CSMTConfig:
    vocab_size: int = 50000
    num_layers: int = 12
    hidden_size: int = 768
    block_size: int = 64
    max_tokens: int = 2048
    num_heads: int = 12
    num_graph_heads: int = 4
    num_experts: int = 8
    moe_top_k: int = 2
    ffn_multiplier: float = 2.0
    kv_compression: float = 0.25
    num_ast_types: int = 256
    ast_dim: int = 128
    ast_gate_scale: float = 0.1
    boundary_mix: float = 0.1
    boundary_width: int = 1
    cvd_prob: float = 0.05
    cvd_scope: str = "variable"
    cvd_audit: bool = False
    dropout: float = 0.0
    validate_input_ranges: bool = True
    tie_embeddings: bool = False
    use_ast_gate: bool = True
    use_block_graph: bool = True
    use_cvd: bool = True
    use_moe: bool = True
    use_boundary: bool = True

    def __post_init__(self) -> None:
        positive_ints = {
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "block_size": self.block_size,
            "max_tokens": self.max_tokens,
            "num_heads": self.num_heads,
            "num_graph_heads": self.num_graph_heads,
            "num_experts": self.num_experts,
            "moe_top_k": self.moe_top_k,
            "num_ast_types": self.num_ast_types,
            "ast_dim": self.ast_dim,
            "boundary_width": self.boundary_width,
        }
        for name, value in positive_ints.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}")
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        float_values = {
            "ffn_multiplier": self.ffn_multiplier,
            "kv_compression": self.kv_compression,
            "ast_gate_scale": self.ast_gate_scale,
            "boundary_mix": self.boundary_mix,
            "cvd_prob": self.cvd_prob,
            "dropout": self.dropout,
        }
        for name, value in float_values.items():
            if not _is_real_number(value):
                raise TypeError(f"{name} must be a finite real number, got {value!r}")
        bool_flags = {
            "tie_embeddings": self.tie_embeddings,
            "use_ast_gate": self.use_ast_gate,
            "use_block_graph": self.use_block_graph,
            "use_cvd": self.use_cvd,
            "cvd_audit": self.cvd_audit,
            "validate_input_ranges": self.validate_input_ranges,
            "use_moe": self.use_moe,
            "use_boundary": self.use_boundary,
        }
        for name, value in bool_flags.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool, got {type(value).__name__}")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.hidden_size % self.num_graph_heads != 0:
            raise ValueError("hidden_size must be divisible by num_graph_heads")
        if self.moe_top_k > self.num_experts:
            raise ValueError("moe_top_k must be <= num_experts")
        if not 0.0 <= self.cvd_prob <= 1.0:
            raise ValueError("cvd_prob must be in [0, 1]")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.ffn_multiplier <= 0:
            raise ValueError("ffn_multiplier must be positive")
        if int(self.hidden_size * self.ffn_multiplier) <= 0:
            raise ValueError("hidden_size * ffn_multiplier must produce at least one FFN channel")
        if not 0.0 < self.kv_compression <= 1.0:
            raise ValueError("kv_compression must be in (0, 1]")
        if not 0.0 <= self.ast_gate_scale < 1.0:
            raise ValueError("ast_gate_scale must be in [0, 1)")
        if self.use_ast_gate and self.num_ast_types < 2:
            raise ValueError("num_ast_types must include at least <PAD> and <UNKNOWN> when use_ast_gate=True")
        if not 0.0 <= self.boundary_mix <= 1.0:
            raise ValueError("boundary_mix must be in [0, 1]")
        if self.boundary_width > self.block_size:
            raise ValueError("boundary_width must be <= block_size")
        if self.cvd_scope not in {"variable", "random"}:
            raise ValueError("cvd_scope must be either 'variable' or 'random'")

    @property
    def num_blocks(self) -> int:
        return math.ceil(self.max_tokens / self.block_size)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        out = x.float() * scale * self.weight.float()
        return out.to(dtype=x.dtype)


def _pad_batch_to_block(x: torch.Tensor, block_size: int) -> torch.Tensor:
    pad = (-x.size(1)) % block_size
    if pad == 0:
        return x
    padding = torch.zeros(x.size(0), pad, dtype=x.dtype, device=x.device)
    return torch.cat([x, padding], dim=1)


def _make_token_mask(lengths: torch.Tensor, padded_length: int, block_size: int) -> torch.Tensor:
    positions = torch.arange(padded_length, device=lengths.device)
    return (positions.view(1, -1) < lengths.view(-1, 1)).view(lengths.numel(), -1, block_size)


class BlockSelfAttention(nn.Module):
    """MLA-style block-local attention with compressed keys and values."""

    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = d // config.num_heads
        if d % config.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        compressed = max(1, int(d * config.kv_compression))
        self.q_proj = nn.Linear(d, d, bias=False)
        self.kv_compress = nn.Linear(d, compressed, bias=False)
        self.k_decompress = nn.Linear(compressed, d, bias=False)
        self.v_decompress = nn.Linear(compressed, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.dropout = config.dropout

    def forward(self, x_blocks: torch.Tensor) -> torch.Tensor:
        n, m, b, d = x_blocks.shape
        h = self.num_heads
        q = self.q_proj(x_blocks).view(n * m, b, h, self.head_dim).transpose(1, 2)
        kv = self.kv_compress(x_blocks)
        k = self.k_decompress(kv).view(n * m, b, h, self.head_dim).transpose(1, 2)
        v = self.v_decompress(kv).view(n * m, b, h, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(n, m, b, d)
        return self.out_proj(y)


class ASTGatedPool(nn.Module):
    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        self.config = config
        self.q_pool = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.ast_gate_scale = config.ast_gate_scale
        self.use_ast_gate = config.use_ast_gate
        self.ast_to_hidden = nn.Linear(config.ast_dim, config.hidden_size, bias=False) if config.use_ast_gate else None
        if self.ast_to_hidden is not None:
            nn.init.normal_(self.ast_to_hidden.weight, std=0.01)

    def forward(
        self,
        h_local: torch.Tensor,
        ast_embeds: Optional[torch.Tensor],
        token_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.use_ast_gate and ast_embeds is not None:
            if self.ast_to_hidden is None:
                raise RuntimeError("AST gate is enabled, but ast_to_hidden was not initialized.")
            gate = torch.tanh(self.ast_to_hidden(ast_embeds.to(dtype=h_local.dtype)))
            gate = 1.0 + self.ast_gate_scale * gate
            h_local = h_local * gate

        scale = math.sqrt(h_local.size(-1))
        scores = torch.einsum("nmbd,d->nmb", h_local.float(), self.q_pool.float()) / scale
        valid_blocks = token_mask.any(dim=-1)
        scores = scores.masked_fill(~token_mask, torch.finfo(scores.dtype).min)
        scores = torch.where(valid_blocks.unsqueeze(-1), scores, torch.zeros_like(scores))
        weights = F.softmax(scores, dim=-1).to(dtype=h_local.dtype)
        weights = weights.masked_fill(~token_mask, 0)
        z = torch.einsum("nmb,nmbd->nmd", weights, h_local)
        return h_local, z


class BoundaryAwarePoolInput(nn.Module):
    """Causal context sharing for tokens near hard block boundaries."""

    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        self.mix = config.boundary_mix
        self.enabled = config.use_boundary
        self.width = config.boundary_width
        out_features = config.hidden_size * config.boundary_width
        self.left_proj = nn.Linear(config.hidden_size, out_features, bias=False) if config.use_boundary else None
        if self.left_proj is not None:
            nn.init.zeros_(self.left_proj.weight)

    def forward(self, h_local: torch.Tensor, token_mask: torch.Tensor) -> torch.Tensor:
        if not self.enabled or self.mix <= 0 or h_local.size(1) <= 1:
            return h_local

        mixed = h_local.clone()
        if self.left_proj is None:
            raise RuntimeError("Boundary mixing is enabled, but left_proj was not initialized.")
        width = min(self.width, h_local.size(2))
        prev_tail = self.left_proj(h_local[:, :-1, -1])
        prev_tail = prev_tail.view(h_local.size(0), h_local.size(1) - 1, self.width, h_local.size(-1))[:, :, :width]
        tail_valid = token_mask[:, :-1, -1]
        head_valid = token_mask[:, 1:, :width]
        valid = head_valid & tail_valid.unsqueeze(-1)
        update = self.mix * prev_tail * valid.unsqueeze(-1).to(dtype=h_local.dtype)
        mixed[:, 1:, :width] = mixed[:, 1:, :width] + update
        return mixed


class PrefixBlockGraph(nn.Module):
    """Autoregressive message passing over block summaries.

    Contextual Variable Dropout (CVD) replaces value messages for selected
    variable-definition blocks. This is a structural regularizer, not a Pearl
    formal intervention claim.
    """

    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        d = config.hidden_size
        h = config.num_graph_heads
        if d % h != 0:
            raise ValueError("hidden_size must be divisible by num_graph_heads")
        self.config = config
        self.num_heads = h
        self.head_dim = d // h
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.out_proj = nn.Linear(d, d, bias=False)
        self.cvd_replacement = nn.Parameter(torch.randn(d) * 0.02) if config.use_cvd else None
        self.last_cvd_audit: Dict[str, float] = {}

    def _clear_cvd_audit(self, valid_block_mask: Optional[torch.Tensor] = None) -> None:
        if not self.config.cvd_audit:
            self.last_cvd_audit = {}
            return
        valid_count = 0.0
        if valid_block_mask is not None:
            valid_count = float(valid_block_mask.to(dtype=torch.bool).sum().detach().cpu().item())
        self.last_cvd_audit = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": valid_count}

    def _record_cvd_audit(
        self,
        eligible: torch.Tensor,
        mask: torch.Tensor,
        valid_block_mask: Optional[torch.Tensor],
    ) -> None:
        if not self.config.cvd_audit:
            self.last_cvd_audit = {}
            return
        valid_count = 0.0
        if valid_block_mask is not None:
            valid_count = float(valid_block_mask.to(dtype=torch.bool).sum().detach().cpu().item())
        eligible_count = float(eligible.sum().detach().cpu().item())
        sampled_count = float(mask.sum().detach().cpu().item())
        self.last_cvd_audit = {
            "eligible_blocks": eligible_count,
            "sampled_blocks": sampled_count,
            "valid_blocks": valid_count,
            "sample_rate": sampled_count / max(1.0, eligible_count),
            "scope_is_variable": 1.0 if self.config.cvd_scope == "variable" else 0.0,
        }

    def _sample_cvd_mask(
        self,
        var_def_mask: Optional[torch.Tensor],
        valid_block_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.config.use_cvd or not self.training or self.config.cvd_prob <= 0:
            self._clear_cvd_audit(valid_block_mask)
            return None
        if self.cvd_replacement is None:
            raise RuntimeError("CVD is enabled, but cvd_replacement was not initialized.")
        if self.config.cvd_scope == "variable":
            if var_def_mask is None:
                self._clear_cvd_audit(valid_block_mask)
                return None
            eligible = var_def_mask.to(dtype=torch.bool, device=self.cvd_replacement.device)
        else:
            if valid_block_mask is None:
                self._clear_cvd_audit(valid_block_mask)
                return None
            eligible = valid_block_mask.to(dtype=torch.bool, device=self.cvd_replacement.device)
        if valid_block_mask is not None:
            valid = valid_block_mask.to(dtype=torch.bool, device=eligible.device)
            if valid.shape != eligible.shape:
                raise ValueError(f"valid_block_mask shape {tuple(valid.shape)} must match eligible mask shape {tuple(eligible.shape)}.")
            eligible = eligible & valid
        mask = (torch.rand_like(eligible, dtype=torch.float32) < self.config.cvd_prob) & eligible
        self._record_cvd_audit(eligible, mask, valid_block_mask)
        return mask

    def forward(
        self,
        z: torch.Tensor,
        var_def_mask: Optional[torch.Tensor],
        valid_block_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n, m, d = z.shape
        valid = None
        if valid_block_mask is not None:
            valid = valid_block_mask.to(device=z.device, dtype=torch.bool)
            if valid.shape != (n, m):
                raise ValueError(f"valid_block_mask shape {tuple(valid.shape)} must match block state shape {(n, m)}.")

        cvd_mask = self._sample_cvd_mask(var_def_mask, valid_block_mask)
        q = self.q_proj(z).view(n, m, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z).view(n, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(n, m, self.num_heads, self.head_dim)
        if cvd_mask is not None:
            if self.cvd_replacement is None:
                raise RuntimeError("CVD mask was sampled, but cvd_replacement is unavailable.")
            replacement = self.cvd_replacement.view(1, self.num_heads, self.head_dim).to(dtype=v.dtype)
            v = torch.where(cvd_mask.view(n, m, 1, 1), replacement, v)
        v = v.transpose(1, 2)
        if valid is None:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            causal = torch.ones(m, m, device=z.device, dtype=torch.bool).tril()
            key_valid = valid.view(n, 1, 1, m)
            attn_mask = causal.view(1, 1, m, m) & key_valid
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, is_causal=False)
            y = y.masked_fill(~valid.view(n, 1, m, 1), 0)
        y = y.transpose(1, 2).contiguous().view(n, m, d)
        out = self.out_proj(y) + z
        if valid is not None:
            out = out.masked_fill(~valid.unsqueeze(-1), 0)
        return out, cvd_mask


class GatedGraphInjection(nn.Module):
    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        d = config.hidden_size
        self.block_size = config.block_size
        self.norm = RMSNorm(d)
        self.gate = nn.Linear(d, d, bias=False)
        self.value = nn.Linear(d, d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        nn.init.normal_(self.gate.weight, std=0.01)
        nn.init.normal_(self.out.weight, std=0.01)

    def forward(self, h_local: torch.Tensor, block_state: torch.Tensor) -> torch.Tensor:
        zero = torch.zeros(
            block_state.size(0),
            1,
            block_state.size(-1),
            dtype=block_state.dtype,
            device=block_state.device,
        )
        prefix = torch.cat([zero, block_state[:, :-1]], dim=1)
        prefix = self.value(self.norm(prefix))
        gates = torch.sigmoid(self.gate(h_local))
        context = gates * prefix.unsqueeze(2)
        return h_local + self.out(context)


class ExpertMLP(nn.Module):
    def __init__(self, hidden_size: int, ffn_size: int) -> None:
        super().__init__()
        self.up = nn.Linear(hidden_size, ffn_size, bias=False)
        self.gate = nn.Linear(hidden_size, ffn_size, bias=False)
        self.down = nn.Linear(ffn_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.up(x)) * self.gate(x))


class SortedTopKMoE(nn.Module):
    """Top-k MoE fallback with sorted contiguous expert batches."""

    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.top_k = config.moe_top_k
        d = config.hidden_size
        ffn = int(d * config.ffn_multiplier)
        self.router = nn.Linear(d, config.num_experts, bias=False)
        self.experts = nn.ModuleList([ExpertMLP(d, ffn) for _ in range(config.num_experts)])
        self.last_load_balance_loss: Optional[torch.Tensor] = None
        self.last_router_z_loss: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, original_shape[-1])
        logits = self.router(x_flat.float())
        probs = F.softmax(logits, dim=-1)
        top_probs, top_idx = torch.topk(probs, k=self.top_k, dim=-1)
        top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        self.last_router_z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()
        expert_assignments = F.one_hot(top_idx, num_classes=self.num_experts).float().mean(dim=(0, 1))
        expert_probs = probs.mean(dim=0)
        self.last_load_balance_loss = self.num_experts * torch.sum(expert_assignments.detach() * expert_probs)

        num_tokens = x_flat.size(0)
        token_ids = torch.arange(num_tokens, device=x_flat.device).repeat_interleave(self.top_k)
        expert_ids = top_idx.reshape(-1)
        weights = top_probs.reshape(-1).to(dtype=x_flat.dtype)

        order = torch.argsort(expert_ids)
        expert_ids = expert_ids.index_select(0, order)
        token_ids = token_ids.index_select(0, order)
        weights = weights.index_select(0, order)

        outputs = torch.zeros_like(x_flat)
        unique_experts, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        for expert_pos, expert_tensor in enumerate(unique_experts.tolist()):
            expert_id = int(expert_tensor)
            start = int(offsets[expert_pos].item())
            end = int(offsets[expert_pos + 1].item())
            ids = token_ids[start:end]
            expert_input = x_flat.index_select(0, ids)
            expert_output = self.experts[expert_id](expert_input)
            outputs.index_add_(0, ids, expert_output * weights[start:end, None])

        return outputs.view(original_shape)


class CSMTLayer(nn.Module):
    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        self.config = config
        self.norm_attn = RMSNorm(config.hidden_size)
        self.attn = BlockSelfAttention(config)
        self.boundary = BoundaryAwarePoolInput(config)
        self.pool = ASTGatedPool(config)
        self.graph = PrefixBlockGraph(config) if config.use_block_graph else None
        self.inject = GatedGraphInjection(config) if config.use_block_graph else None
        self.norm_moe = RMSNorm(config.hidden_size)
        self.use_moe = config.use_moe
        ffn_size = int(config.hidden_size * config.ffn_multiplier)
        self.moe = SortedTopKMoE(config) if config.use_moe else None
        self.dense_ffn = None if config.use_moe else ExpertMLP(config.hidden_size, ffn_size)

    def forward(
        self,
        x_blocks: torch.Tensor,
        ast_embeds: Optional[torch.Tensor],
        token_mask: torch.Tensor,
        var_def_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        attn_out = self.attn(self.norm_attn(x_blocks))
        h_local = x_blocks + attn_out
        h_local = self.boundary(h_local, token_mask)
        h_local, z = self.pool(h_local, ast_embeds, token_mask)
        if self.config.use_block_graph:
            if self.graph is None or self.inject is None:
                raise RuntimeError("Block graph is enabled, but graph modules were not initialized.")
            block_state, cvd_mask = self.graph(z, var_def_mask, token_mask.any(dim=-1))
            injected = self.inject(h_local, block_state)
        else:
            block_state = z
            cvd_mask = None
            injected = h_local

        ffn_input = self.norm_moe(injected)
        if self.use_moe:
            if self.moe is None:
                raise RuntimeError("MoE is enabled, but the MoE module was not initialized.")
            ffn_out = self.moe(ffn_input)
        else:
            if self.dense_ffn is None:
                raise RuntimeError("MoE is disabled, but dense_ffn was not initialized.")
            ffn_out = self.dense_ffn(ffn_input)
        out = injected + ffn_out
        return out, block_state, cvd_mask


class CSMTModel(nn.Module):
    def __init__(self, config: CSMTConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.token_embed = nn.Embedding(config.vocab_size, d)
        self.pos_embed = nn.Embedding(config.max_tokens, d)
        self.ast_embed = nn.Embedding(config.num_ast_types, config.ast_dim) if config.use_ast_gate else None
        self.layers = nn.ModuleList([CSMTLayer(config) for _ in range(config.num_layers)])
        self.final_norm = RMSNorm(d)
        self.output_head = nn.Linear(d, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.output_head.weight = self.token_embed.weight

    def forward(
        self,
        input_ids: torch.Tensor,
        ast_type_ids: Optional[torch.Tensor] = None,
        var_def_mask: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        squeeze_output = False
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
            squeeze_output = True
        elif input_ids.dim() != 2:
            raise ValueError("CSMTModel expects input_ids shaped [L] or [batch, L].")

        input_ids = input_ids[:, : self.config.max_tokens]
        _validate_id_tensor("input_ids", input_ids, self.config.vocab_size, self.config.validate_input_ranges)
        input_ids = input_ids.long()
        batch_size, length = input_ids.shape
        if length == 0:
            raise ValueError("input_ids must contain at least one token after max_tokens truncation.")
        if lengths is None:
            lengths = torch.full((batch_size,), length, dtype=torch.long, device=input_ids.device)
        else:
            _validate_length_tensor("lengths", lengths)
            lengths = lengths.to(device=input_ids.device).view(-1).long()
            if lengths.numel() != batch_size:
                raise ValueError(f"lengths must have batch dimension {batch_size}, got {lengths.numel()}.")
            if bool((lengths < 0).any()) or bool((lengths > length).any()):
                min_len = int(lengths.min().item())
                max_len = int(lengths.max().item())
                raise ValueError(f"lengths must be in [0, {length}] after max_tokens truncation, got min={min_len}, max={max_len}.")
        padded_ids = _pad_batch_to_block(input_ids, self.config.block_size)
        padded_length = padded_ids.size(1)
        num_blocks = padded_length // self.config.block_size
        positions = torch.arange(padded_length, device=input_ids.device).clamp_max(self.config.max_tokens - 1)
        x = self.token_embed(padded_ids) + self.pos_embed(positions).unsqueeze(0)
        x = x.view(batch_size, num_blocks, self.config.block_size, self.config.hidden_size)
        token_mask = _make_token_mask(lengths, padded_length, self.config.block_size)

        ast_embeds = None
        if ast_type_ids is not None and self.config.use_ast_gate:
            if self.ast_embed is None:
                raise RuntimeError("AST gate is enabled, but ast_embed was not initialized.")
            if ast_type_ids.dim() == 2:
                ast_type_ids = ast_type_ids.unsqueeze(0)
                if batch_size > 1:
                    ast_type_ids = ast_type_ids.expand(batch_size, -1, -1)
            if ast_type_ids.dim() != 3:
                raise ValueError("ast_type_ids must have shape [num_blocks, block_size] or [batch, num_blocks, block_size].")
            if ast_type_ids.size(0) != batch_size:
                raise ValueError(f"ast_type_ids batch dimension must be {batch_size}, got {ast_type_ids.size(0)}.")
            if ast_type_ids.size(2) != self.config.block_size:
                raise ValueError(
                    f"ast_type_ids last dimension must equal block_size={self.config.block_size}, "
                    f"got {ast_type_ids.size(2)}."
                )
            ast_type_ids = ast_type_ids[:, :num_blocks].to(device=input_ids.device)
            _validate_id_tensor(
                "ast_type_ids",
                ast_type_ids,
                self.config.num_ast_types,
                self.config.validate_input_ranges,
            )
            ast_type_ids = ast_type_ids.long()
            if ast_type_ids.size(1) < num_blocks:
                pad_blocks = num_blocks - ast_type_ids.size(1)
                pad = torch.zeros(
                    batch_size,
                    pad_blocks,
                    self.config.block_size,
                    device=input_ids.device,
                    dtype=torch.long,
                )
                ast_type_ids = torch.cat([ast_type_ids, pad], dim=1)
            ast_embeds = self.ast_embed(ast_type_ids)

        if var_def_mask is not None and self.config.use_block_graph and self.config.use_cvd:
            _validate_mask_tensor("var_def_mask", var_def_mask)
            if var_def_mask.dim() == 1:
                var_def_mask = var_def_mask.unsqueeze(0)
                if batch_size > 1:
                    var_def_mask = var_def_mask.expand(batch_size, -1)
            if var_def_mask.dim() != 2:
                raise ValueError("var_def_mask must have shape [num_blocks], [batch, num_blocks], or [batch, tokens].")
            if var_def_mask.size(0) != batch_size:
                raise ValueError(f"var_def_mask batch dimension must be {batch_size}, got {var_def_mask.size(0)}.")

            mask_width = var_def_mask.size(1)
            looks_token_level = mask_width > num_blocks or mask_width == length
            if looks_token_level:
                token_level = var_def_mask[:, :length].to(device=input_ids.device, dtype=torch.bool)
                if token_level.size(1) < length:
                    pad = torch.zeros(batch_size, length - token_level.size(1), device=input_ids.device, dtype=torch.bool)
                    token_level = torch.cat([token_level, pad], dim=1)
                token_level = _pad_batch_to_block(token_level, self.config.block_size)
                var_def_mask = token_level.view(batch_size, num_blocks, self.config.block_size).any(dim=-1)
            else:
                var_def_mask = var_def_mask[:, :num_blocks].to(device=input_ids.device, dtype=torch.bool)
                if var_def_mask.size(1) < num_blocks:
                    pad = torch.zeros(batch_size, num_blocks - var_def_mask.size(1), device=input_ids.device, dtype=torch.bool)
                    var_def_mask = torch.cat([var_def_mask, pad], dim=1)

        for layer in self.layers:
            x, _, _ = layer(x, ast_embeds, token_mask, var_def_mask)
            x = x.masked_fill(~token_mask[..., None], 0)

        x = x.view(batch_size, padded_length, self.config.hidden_size)[:, :length]
        logits = self.output_head(self.final_norm(x))
        return logits.squeeze(0) if squeeze_output else logits

    def cvd_regularization(self) -> torch.Tensor:
        if not self.config.use_block_graph or not self.config.use_cvd:
            return torch.tensor(0.0, device=self.token_embed.weight.device)
        reg = None
        for layer in self.layers:
            if layer.graph is None or layer.graph.cvd_replacement is None:
                continue
            term = layer.graph.cvd_replacement.float().pow(2).mean()
            reg = term if reg is None else reg + term
        if reg is None:
            return torch.tensor(0.0, device=self.token_embed.weight.device)
        return reg / max(1, len(self.layers))

    def moe_auxiliary_loss(self) -> torch.Tensor:
        if not self.config.use_moe:
            return torch.tensor(0.0, device=self.token_embed.weight.device)
        losses = []
        for layer in self.layers:
            if layer.moe is not None and layer.moe.last_load_balance_loss is not None:
                losses.append(layer.moe.last_load_balance_loss)
        if not losses:
            return torch.tensor(0.0, device=self.token_embed.weight.device)
        return torch.stack(losses).mean()

    def router_z_loss(self) -> torch.Tensor:
        if not self.config.use_moe:
            return torch.tensor(0.0, device=self.token_embed.weight.device)
        losses = []
        for layer in self.layers:
            if layer.moe is not None and layer.moe.last_router_z_loss is not None:
                losses.append(layer.moe.last_router_z_loss)
        if not losses:
            return torch.tensor(0.0, device=self.token_embed.weight.device)
        return torch.stack(losses).mean()

    def cvd_audit_summary(self) -> Dict[str, float]:
        if not self.config.use_block_graph:
            return {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": 0.0, "layers_with_cvd": 0.0}
        totals = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": 0.0, "layers_with_cvd": 0.0}
        for layer in self.layers:
            if layer.graph is None:
                continue
            audit = layer.graph.last_cvd_audit
            totals["eligible_blocks"] += float(audit.get("eligible_blocks", 0.0))
            totals["sampled_blocks"] += float(audit.get("sampled_blocks", 0.0))
            totals["valid_blocks"] += float(audit.get("valid_blocks", 0.0))
            if audit:
                totals["layers_with_cvd"] += 1.0
        totals["sample_rate"] = totals["sampled_blocks"] / max(1.0, totals["eligible_blocks"])
        return totals
