"""
Tiny falsification-oriented training run for CSMT-GNN.

This script is intentionally small. It uses the synthetic structural cases from
diagnostics.py and trains a matrix of tiny ablations for a few steps:

1. transformer_baseline: a plain token-only causal Transformer;
2. transformer_matched: a token-only causal Transformer with a wider FFN,
   used as a rough parameter-neighbor control for the CSMT variants;
3. token_baseline: the CSMT local-token path with dense FFN;
4. ast_only: token model plus residual AST gate;
5. graph_only: token model plus prefix block graph;
6. ast_graph: AST gate and prefix block graph, without CVD;
7. random_dropout_control: AST+graph with random block value replacement;
8. variable_cvd: AST+graph with variable-definition CVD;
9. full_moe: variable CVD plus the sorted top-k MoE fallback.

The output is a JSON file with loss traces and environment metadata. It is not a
benchmark result. It is a cheap sanity check that the plumbing can be trained
before spending real compute.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics import run as build_diagnostics

try:
    import torch
    import torch.nn.functional as F
except ImportError as exc:  # pragma: no cover - this path is for user machines without torch.
    raise SystemExit("PyTorch is required for diagnostic_poc_train.py. Install torch first.") from exc

from csmt_gnn import CSMTConfig, CSMTModel
from train import trim_features_for_next_token
from transformer_baseline import TinyCausalTransformer, TransformerBaselineConfig


@dataclass
class PocResult:
    variant: str
    losses: List[float]
    final_loss: float
    loss_delta: float
    parameter_count: int
    trainable_parameter_count: int
    config: Dict
    eval_metrics: Dict[str, float]


class DiagnosticArrays:
    def __init__(self, root: Path, max_tokens: int) -> None:
        self.token_dir = root / "tokens"
        self.ast_dir = root / "ast"
        self.max_tokens = max_tokens
        self.prefixes = sorted(path.name[: -len("_tokens.npy")] for path in self.token_dir.glob("*_tokens.npy"))
        if not self.prefixes:
            raise RuntimeError(f"No diagnostic token arrays found in {self.token_dir}")

    def load(self, prefix: str) -> Dict[str, torch.Tensor]:
        tokens = np.load(self.token_dir / f"{prefix}_tokens.npy")
        ast_ids = np.load(self.ast_dir / f"{prefix}_ast_ids.npy")
        token_mask = np.load(self.ast_dir / f"{prefix}_token_ast_mask.npy")
        return {
            "tokens": torch.from_numpy(np.asarray(tokens[: self.max_tokens], dtype=np.int64)),
            "ast_ids": torch.from_numpy(np.asarray(ast_ids, dtype=np.int64)),
            "ast_mask": torch.from_numpy(np.asarray(token_mask, dtype=np.bool_)),
        }


def load_vocab_size(root: Path) -> int:
    vocab_path = root / "tokens" / "token_vocab.json"
    if not vocab_path.exists():
        return 128
    vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    return max(8, len(vocab))


def load_num_ast_types(root: Path) -> int:
    vocab_path = root / "ast" / "ast_vocab.json"
    array_max = -1
    for path in (root / "ast").glob("*_ast_ids.npy"):
        values = np.load(path)
        if values.size:
            array_max = max(array_max, int(values.max()))
    if not vocab_path.exists():
        return max(64, array_max + 1)
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    type_vocab = data.get("type_vocab", {})
    if not type_vocab:
        return max(64, array_max + 1)
    metadata_size = max(int(idx) for idx in type_vocab.values()) + 1
    return max(8, metadata_size, array_max + 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_config(args, variant: str, vocab_size: int, num_ast_types: int) -> CSMTConfig:
    base = dict(
        vocab_size=vocab_size,
        num_layers=1,
        hidden_size=args.hidden_size,
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        num_heads=4,
        num_graph_heads=4,
        num_experts=1,
        moe_top_k=1,
        ffn_multiplier=1.5,
        kv_compression=0.5,
        num_ast_types=num_ast_types,
        ast_dim=args.ast_dim,
        ast_gate_scale=args.ast_gate_scale,
        boundary_mix=args.boundary_mix,
        boundary_width=args.boundary_width,
        cvd_prob=0.0,
        cvd_scope="variable",
        dropout=0.0,
        use_ast_gate=False,
        use_block_graph=False,
        use_cvd=False,
        use_moe=False,
        use_boundary=False,
    )
    variants = {
        "token_baseline": {},
        "boundary_only": {"use_boundary": True},
        "ast_only": {"use_ast_gate": True},
        "graph_only": {"use_block_graph": True},
        "ast_graph": {"use_ast_gate": True, "use_block_graph": True, "use_boundary": True},
        "random_dropout_control": {
            "use_ast_gate": True,
            "use_block_graph": True,
            "use_cvd": True,
            "use_boundary": True,
            "cvd_prob": args.cvd_prob,
            "cvd_scope": "random",
        },
        "variable_cvd": {
            "use_ast_gate": True,
            "use_block_graph": True,
            "use_cvd": True,
            "use_boundary": True,
            "cvd_prob": args.cvd_prob,
            "cvd_scope": "variable",
        },
        "full_moe": {
            "use_ast_gate": True,
            "use_block_graph": True,
            "use_cvd": True,
            "use_moe": True,
            "use_boundary": True,
            "num_experts": 2,
            "cvd_prob": args.cvd_prob,
            "cvd_scope": "variable",
        },
    }
    if variant not in variants:
        raise ValueError(f"Unknown diagnostic variant: {variant}")
    base.update(variants[variant])
    return CSMTConfig(**base)


def parameter_counts(model: torch.nn.Module) -> Dict[str, int]:
    params = list(model.parameters())
    return {
        "parameter_count": sum(param.numel() for param in params),
        "trainable_parameter_count": sum(param.numel() for param in params if param.requires_grad),
    }


def make_transformer_config(args, variant: str, vocab_size: int) -> TransformerBaselineConfig:
    ffn_multiplier = 4.5 if variant == "transformer_matched" else 1.5
    return TransformerBaselineConfig(
        vocab_size=vocab_size,
        num_layers=1,
        hidden_size=args.hidden_size,
        max_tokens=args.max_tokens,
        num_heads=4,
        ffn_multiplier=ffn_multiplier,
        dropout=0.0,
    )


def is_transformer_variant(variant: str) -> bool:
    return variant in {"transformer_baseline", "transformer_matched"}


def build_model(args, variant: str, vocab_size: int, num_ast_types: int):
    if is_transformer_variant(variant):
        config = make_transformer_config(args, variant, vocab_size)
        return TinyCausalTransformer(config), config
    config = make_config(args, variant, vocab_size, num_ast_types)
    return CSMTModel(config), config


def trim_batch_to_prefix(batch: Dict[str, torch.Tensor], input_length: int, block_size: int) -> Dict[str, torch.Tensor]:
    """Return a batch copy whose structural side inputs end at the model prefix."""

    ast_ids = batch["ast_ids"]
    ast_mask = batch["ast_mask"]
    ast_was_unbatched = ast_ids.dim() == 2
    mask_was_unbatched = ast_mask.dim() == 1
    if ast_was_unbatched:
        ast_ids = ast_ids.unsqueeze(0)
    if mask_was_unbatched:
        ast_mask = ast_mask.unsqueeze(0)
    ast_ids, ast_mask = trim_features_for_next_token(ast_ids, ast_mask, input_length, block_size)
    trimmed = dict(batch)
    trimmed["ast_ids"] = ast_ids.squeeze(0) if ast_was_unbatched else ast_ids
    trimmed["ast_mask"] = ast_mask.squeeze(0) if mask_was_unbatched else ast_mask
    return trimmed


def audit_prefix_feature_trimming(arrays: DiagnosticArrays, block_size: int) -> Dict[str, object]:
    """Summarize whether diagnostic AST side inputs are prefix-aligned."""

    violations: List[Dict[str, object]] = []
    cases = 0
    token_level_masks = 0
    max_input_length = 0
    max_ast_blocks = 0
    max_mask_width = 0
    for prefix in arrays.prefixes:
        batch = arrays.load(prefix)
        tokens = batch["tokens"]
        if tokens.numel() < 2:
            continue
        cases += 1
        input_length = int(tokens.numel() - 1)
        required_blocks = (input_length + block_size - 1) // block_size
        original_blocks = int(batch["ast_ids"].size(0))
        full_mask_width = int(batch["ast_mask"].numel())
        mask_is_token_level = full_mask_width > original_blocks
        if mask_is_token_level:
            token_level_masks += 1
        expected_mask_width = input_length if mask_is_token_level else required_blocks
        trimmed = trim_batch_to_prefix(batch, input_length, block_size)
        ast_blocks = int(trimmed["ast_ids"].size(0))
        mask_width = int(trimmed["ast_mask"].numel())
        max_input_length = max(max_input_length, input_length)
        max_ast_blocks = max(max_ast_blocks, ast_blocks)
        max_mask_width = max(max_mask_width, mask_width)
        if ast_blocks != required_blocks or mask_width != expected_mask_width:
            violations.append(
                {
                    "prefix": prefix,
                    "input_length": input_length,
                    "required_blocks": required_blocks,
                    "ast_blocks": ast_blocks,
                    "mask_width": mask_width,
                    "expected_mask_width": expected_mask_width,
                }
            )
    return {
        "cases": cases,
        "token_level_masks": token_level_masks,
        "max_input_length": max_input_length,
        "max_ast_blocks": max_ast_blocks,
        "max_mask_width": max_mask_width,
        "all_prefix_aligned": float(len(violations) == 0),
        "violations": violations,
    }


def forward_variant(model, variant: str, batch: Dict[str, torch.Tensor], input_ids: torch.Tensor) -> torch.Tensor:
    lengths = torch.tensor([input_ids.numel()], dtype=torch.long)
    if is_transformer_variant(variant):
        return model(input_ids, lengths=lengths)
    batch = trim_batch_to_prefix(batch, input_ids.numel(), model.config.block_size)
    return model(
        input_ids,
        ast_type_ids=batch["ast_ids"],
        var_def_mask=batch["ast_mask"],
        lengths=lengths,
    )


def dependency_target_masks(tokens: torch.Tensor, definition_mask: torch.Tensor, block_size: int) -> Dict[str, torch.Tensor]:
    """Return target-position masks for simple definition-use probes.

    For next-token loss, target position j predicts source token j+1.  The masks
    below are therefore shifted from source-token positions into label positions.
    """

    source_len = int(tokens.numel())
    use = torch.zeros(source_len, dtype=torch.bool)
    cross_block = torch.zeros(source_len, dtype=torch.bool)
    first_definition: Dict[int, int] = {}
    for idx, token_id in enumerate(tokens.tolist()):
        is_definition = idx < definition_mask.numel() and bool(definition_mask[idx])
        if is_definition:
            first_definition[token_id] = idx
            continue
        def_idx = first_definition.get(token_id)
        if def_idx is None or idx <= def_idx:
            continue
        use[idx] = True
        cross_block[idx] = (idx // block_size) > (def_idx // block_size)
    return {"use": use[1:], "cross_block_use": cross_block[1:]}


def masked_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    per_token = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), reduction="none")
    return float(per_token[mask.view(-1)].mean().detach().cpu().item())


def masked_top1_rate(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return float("nan")
    predictions = logits.argmax(dim=-1).view(-1)
    flat_labels = labels.view(-1)
    flat_mask = mask.view(-1)
    return float((predictions[flat_mask] == flat_labels[flat_mask]).float().mean().detach().cpu().item())


@torch.no_grad()
def evaluate_variant(args, model, variant: str, arrays: DiagnosticArrays) -> Dict[str, float]:
    was_training = model.training
    model.eval()
    total_losses: List[float] = []
    use_losses: List[float] = []
    cross_block_losses: List[float] = []
    use_top1_rates: List[float] = []
    cross_block_top1_rates: List[float] = []
    use_targets = 0
    cross_block_targets = 0
    for prefix in arrays.prefixes:
        batch = arrays.load(prefix)
        tokens = batch["tokens"]
        if tokens.numel() < 2:
            continue
        input_ids = tokens[:-1]
        labels = tokens[1:]
        logits = forward_variant(model, variant, batch, input_ids)
        total_losses.append(float(F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1)).cpu().item()))
        masks = dependency_target_masks(tokens, batch["ast_mask"], args.block_size)
        use_targets += int(masks["use"].sum().item())
        cross_block_targets += int(masks["cross_block_use"].sum().item())
        use_loss = masked_loss(logits, labels, masks["use"])
        cross_loss = masked_loss(logits, labels, masks["cross_block_use"])
        use_top1 = masked_top1_rate(logits, labels, masks["use"])
        cross_top1 = masked_top1_rate(logits, labels, masks["cross_block_use"])
        if not np.isnan(use_loss):
            use_losses.append(use_loss)
            use_top1_rates.append(use_top1)
        if not np.isnan(cross_loss):
            cross_block_losses.append(cross_loss)
            cross_block_top1_rates.append(cross_top1)
    if was_training:
        model.train()
    return {
        "eval_loss": float(np.mean(total_losses)) if total_losses else float("nan"),
        "definition_use_loss": float(np.mean(use_losses)) if use_losses else float("nan"),
        "cross_block_use_loss": float(np.mean(cross_block_losses)) if cross_block_losses else float("nan"),
        "definition_use_preservation": float(np.mean(use_top1_rates)) if use_top1_rates else float("nan"),
        "cross_block_use_preservation": float(np.mean(cross_block_top1_rates)) if cross_block_top1_rates else float("nan"),
        "definition_use_targets": float(use_targets),
        "cross_block_use_targets": float(cross_block_targets),
    }


def train_variant(args, arrays: DiagnosticArrays, variant: str, vocab_size: int, num_ast_types: int) -> PocResult:
    set_seed(args.seed)
    model, config = build_model(args, variant, vocab_size, num_ast_types)
    model.train()
    counts = parameter_counts(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.0)
    losses: List[float] = []

    for step in range(args.steps):
        prefix = arrays.prefixes[step % len(arrays.prefixes)]
        batch = arrays.load(prefix)
        tokens = batch["tokens"]
        if tokens.numel() < 2:
            continue
        input_ids = tokens[:-1]
        labels = tokens[1:]
        logits = forward_variant(model, variant, batch, input_ids)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1))
        if isinstance(config, CSMTConfig) and config.use_moe:
            loss = loss + 1e-2 * model.moe_auxiliary_loss() + 1e-3 * model.router_z_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))

    final_loss = losses[-1] if losses else float("nan")
    first_loss = losses[0] if losses else float("nan")
    eval_metrics = evaluate_variant(args, model, variant, arrays)
    return PocResult(
        variant=variant,
        losses=losses,
        final_loss=final_loss,
        loss_delta=final_loss - first_loss if losses else float("nan"),
        parameter_count=counts["parameter_count"],
        trainable_parameter_count=counts["trainable_parameter_count"],
        config=asdict(config),
        eval_metrics=eval_metrics,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a tiny CSMT-GNN diagnostic training sanity check.")
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/diagnostic_poc"))
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic_poc_transformer.json"))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--case-set", choices=("tiny", "long", "all"), default="tiny")
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--ast-dim", type=int, default=16)
    parser.add_argument("--ast-gate-scale", type=float, default=0.1)
    parser.add_argument("--boundary-mix", type=float, default=0.1)
    parser.add_argument("--boundary-width", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--cvd-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--variants",
        default="transformer_baseline,transformer_matched,token_baseline,boundary_only,ast_only,graph_only,ast_graph,random_dropout_control,variable_cvd,full_moe",
        help="Comma-separated diagnostic variants to train.",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_diagnostics(args.work_dir, block_size=args.block_size, max_tokens=args.max_tokens, case_set=args.case_set)

    arrays = DiagnosticArrays(args.work_dir, args.max_tokens)
    vocab_size = load_vocab_size(args.work_dir)
    num_ast_types = load_num_ast_types(args.work_dir)
    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    results = {
        "purpose": "tiny plumbing and falsification sanity check; not a benchmark",
        "seed": args.seed,
        "case_set": args.case_set,
        "num_cases": len(arrays.prefixes),
        "vocab_size": vocab_size,
        "num_ast_types": num_ast_types,
        "prefix_feature_audit": audit_prefix_feature_trimming(arrays, args.block_size),
        "variants": [asdict(train_variant(args, arrays, variant, vocab_size, num_ast_types)) for variant in variants],
    }
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "variants": results["variants"]}, indent=2))


if __name__ == "__main__":
    main()
