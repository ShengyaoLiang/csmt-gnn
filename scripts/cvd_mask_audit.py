"""
Audit Contextual Variable Dropout sampling on tiny diagnostics.

The script measures how many blocks are eligible and sampled under
`cvd_scope=variable` and `cvd_scope=random`. It is a mechanism check, not a model
quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diagnostics import run as build_diagnostics
from scripts.diagnostic_poc_train import DiagnosticArrays, load_num_ast_types, load_vocab_size

try:
    import torch
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyTorch is required for cvd_mask_audit.py.") from exc

from csmt_gnn import CSMTConfig, CSMTModel


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_model(args, scope: str, vocab_size: int, num_ast_types: int) -> CSMTModel:
    config = CSMTConfig(
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
        ast_gate_scale=0.1,
        boundary_mix=0.1,
        boundary_width=1,
        cvd_prob=args.cvd_prob,
        cvd_scope=scope,
        use_ast_gate=True,
        use_block_graph=True,
        use_cvd=True,
        use_moe=False,
        use_boundary=True,
    )
    model = CSMTModel(config)
    model.train()
    return model


def audit_scope(args, arrays: DiagnosticArrays, scope: str, vocab_size: int, num_ast_types: int) -> Dict[str, object]:
    set_seed(args.seed)
    model = make_model(args, scope, vocab_size, num_ast_types)
    totals = {"eligible_blocks": 0.0, "sampled_blocks": 0.0, "valid_blocks": 0.0, "layers_with_cvd": 0.0}
    per_step: List[Dict[str, float]] = []
    for step in range(args.steps):
        prefix = arrays.prefixes[step % len(arrays.prefixes)]
        batch = arrays.load(prefix)
        tokens = batch["tokens"]
        if tokens.numel() < 2:
            continue
        input_ids = tokens[:-1].unsqueeze(0)
        with torch.no_grad():
            model(
                input_ids,
                ast_type_ids=batch["ast_ids"].unsqueeze(0),
                var_def_mask=batch["ast_mask"].unsqueeze(0),
                lengths=torch.tensor([input_ids.numel()], dtype=torch.long),
            )
        audit = model.cvd_audit_summary()
        per_step.append({key: float(value) for key, value in audit.items()})
        for key in totals:
            totals[key] += float(audit.get(key, 0.0))
    totals["sample_rate"] = totals["sampled_blocks"] / max(1.0, totals["eligible_blocks"])
    return {"scope": scope, "totals": totals, "per_step": per_step}


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit CVD mask sampling on tiny diagnostics.")
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/cvd_mask_audit"))
    parser.add_argument("--output", type=Path, default=Path("results/cvd_mask_audit.json"))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--ast-dim", type=int, default=8)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--cvd-prob", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    build_diagnostics(args.work_dir, block_size=args.block_size, max_tokens=args.max_tokens)
    arrays = DiagnosticArrays(args.work_dir, args.max_tokens)
    vocab_size = load_vocab_size(args.work_dir)
    num_ast_types = load_num_ast_types(args.work_dir)
    result = {
        "purpose": "CVD sampling audit; not a benchmark",
        "seed": args.seed,
        "cvd_prob": args.cvd_prob,
        "steps": args.steps,
        "scopes": [
            audit_scope(args, arrays, "variable", vocab_size, num_ast_types),
            audit_scope(args, arrays, "random", vocab_size, num_ast_types),
        ],
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
