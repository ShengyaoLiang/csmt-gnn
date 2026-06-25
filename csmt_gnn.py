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
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


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
    dropout: float = 0.0
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
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
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
        if self.kv_compression <= 0:
            raise ValueError("kv_compression must be positive")
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
            assert self.ast_to_hidden is not None
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
        assert self.left_proj is not None
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

    def _sample_cvd_mask(
        self,
        var_def_mask: Optional[torch.Tensor],
        valid_block_mask: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if not self.config.use_cvd or not self.training or self.config.cvd_prob <= 0:
            self.last_cvd_audit = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": 0.0}
            return None
        assert self.cvd_replacement is not None
        valid_count = float(valid_block_mask.to(dtype=torch.bool).sum().item()) if valid_block_mask is not None else 0.0
        if self.config.cvd_scope == "variable":
            if var_def_mask is None:
                self.last_cvd_audit = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": valid_count}
                return None
            eligible = var_def_mask.to(dtype=torch.bool, device=self.cvd_replacement.device)
        else:
            if valid_block_mask is None:
                self.last_cvd_audit = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": 0.0}
                return None
            eligible = valid_block_mask.to(dtype=torch.bool, device=self.cvd_replacement.device)
        if not bool(eligible.any()):
            self.last_cvd_audit = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": valid_count}
            return None
        mask = (torch.rand_like(eligible, dtype=torch.float32) < self.config.cvd_prob) & eligible
        eligible_count = float(eligible.sum().item())
        sampled_count = float(mask.sum().item())
        self.last_cvd_audit = {
            "eligible_blocks": eligible_count,
            "sampled_blocks": sampled_count,
            "valid_blocks": valid_count,
            "sample_rate": sampled_count / max(1.0, eligible_count),
            "scope_is_variable": 1.0 if self.config.cvd_scope == "variable" else 0.0,
        }
        return mask if bool(mask.any()) else None

    def forward(
        self,
        z: torch.Tensor,
        var_def_mask: Optional[torch.Tensor],
        valid_block_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        n, m, d = z.shape
        cvd_mask = self._sample_cvd_mask(var_def_mask, valid_block_mask)
        q = self.q_proj(z).view(n, m, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(z).view(n, m, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(z).view(n, m, self.num_heads, self.head_dim)
        if cvd_mask is not None:
            assert self.cvd_replacement is not None
            replacement = self.cvd_replacement.view(1, self.num_heads, self.head_dim).to(dtype=v.dtype)
            v = torch.where(cvd_mask.view(n, m, 1, 1), replacement, v)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(n, m, d)
        return self.out_proj(y) + z, cvd_mask


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
            assert self.graph is not None
            assert self.inject is not None
            block_state, cvd_mask = self.graph(z, var_def_mask, token_mask.any(dim=-1))
            injected = self.inject(h_local, block_state)
        else:
            block_state = z
            cvd_mask = None
            injected = h_local

        ffn_input = self.norm_moe(injected)
        if self.use_moe:
            assert self.moe is not None
            ffn_out = self.moe(ffn_input)
        else:
            assert self.dense_ffn is not None
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
        batch_size, length = input_ids.shape
        if lengths is None:
            lengths = torch.full((batch_size,), length, dtype=torch.long, device=input_ids.device)
        else:
            lengths = lengths.to(device=input_ids.device, dtype=torch.long).view(-1)
            if lengths.numel() != batch_size:
                raise ValueError(f"lengths must have batch dimension {batch_size}, got {lengths.numel()}.")
            lengths = lengths.clamp(min=0, max=length)
        padded_ids = _pad_batch_to_block(input_ids, self.config.block_size)
        padded_length = padded_ids.size(1)
        num_blocks = padded_length // self.config.block_size
        positions = torch.arange(padded_length, device=input_ids.device).clamp_max(self.config.max_tokens - 1)
        x = self.token_embed(padded_ids) + self.pos_embed(positions).unsqueeze(0)
        x = x.view(batch_size, num_blocks, self.config.block_size, self.config.hidden_size)
        token_mask = _make_token_mask(lengths, padded_length, self.config.block_size)

        ast_embeds = None
        if ast_type_ids is not None and self.config.use_ast_gate:
            assert self.ast_embed is not None
            if ast_type_ids.dim() == 2:
                ast_type_ids = ast_type_ids.unsqueeze(0)
            if ast_type_ids.dim() != 3:
                raise ValueError("ast_type_ids must have shape [num_blocks, block_size] or [batch, num_blocks, block_size].")
            if ast_type_ids.size(0) != batch_size:
                raise ValueError(f"ast_type_ids batch dimension must be {batch_size}, got {ast_type_ids.size(0)}.")
            if ast_type_ids.size(2) != self.config.block_size:
                raise ValueError(
                    f"ast_type_ids last dimension must equal block_size={self.config.block_size}, "
                    f"got {ast_type_ids.size(2)}."
                )
            ast_type_ids = ast_type_ids[:, :num_blocks].to(device=input_ids.device, dtype=torch.long)
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
            ast_embeds = self.ast_embed(ast_type_ids.clamp_max(self.config.num_ast_types - 1))

        if var_def_mask is not None and self.config.use_block_graph and self.config.use_cvd:
            if var_def_mask.dim() == 1:
                var_def_mask = var_def_mask.unsqueeze(0)
            if var_def_mask.dim() != 2:
                raise ValueError("var_def_mask must have shape [num_blocks], [batch, num_blocks], or [batch, tokens].")
            if var_def_mask.size(0) != batch_size:
                raise ValueError(f"var_def_mask batch dimension must be {batch_size}, got {var_def_mask.size(0)}.")

            mask_width = var_def_mask.size(1)
            looks_token_level = mask_width > num_blocks or mask_width == length
            if looks_token_level:
                token_level = var_def_mask[:, :length].to(device=input_ids.device, dtype=torch.bool)
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
