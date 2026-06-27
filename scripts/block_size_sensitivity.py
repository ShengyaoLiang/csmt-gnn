"""Run a small block-size sensitivity diagnostic.

This script is a low-compute companion to the paper's block-size propositions.
It repeats structural coverage and tiny training diagnostics for several block
sizes, then writes a compact summary.  The default sizes are 32, 64, and 128,
matching the paper-facing sensitivity question.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from statistics import mean
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]


def parse_int_list(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one block size")
    return values


def run_command(command: List[str]) -> None:
    subprocess.run(command, cwd=ROOT, check=True)


def variant_by_name(path: Path) -> Dict[str, Dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return {item["variant"]: item for item in data["variants"]}


def structural_coverage(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = data["cases"]
    return {
        "definition_use_pairs": float(sum(case["definition_use_pairs"] for case in cases)),
        "cross_block_pairs": float(sum(case["cross_block_pairs"] for case in cases)),
        "mean_max_block_distance": float(mean(case["max_block_distance"] for case in cases)),
    }


def row_for(block_size: int, probe_path: Path, train_path: Path) -> Dict[str, object]:
    variants = variant_by_name(train_path)
    coverage = structural_coverage(probe_path)
    ast_graph = variants.get("ast_graph", {})
    transformer = variants.get("transformer_matched", {})
    return {
        "block_size": block_size,
        **coverage,
        "ast_graph_cross_block_use_loss": ast_graph.get("eval_metrics", {}).get("cross_block_use_loss"),
        "transformer_matched_cross_block_use_loss": transformer.get("eval_metrics", {}).get("cross_block_use_loss"),
        "ast_graph_cross_block_use_preservation": ast_graph.get("eval_metrics", {}).get("cross_block_use_preservation"),
        "transformer_matched_cross_block_use_preservation": transformer.get("eval_metrics", {}).get(
            "cross_block_use_preservation"
        ),
        "ast_graph_params": ast_graph.get("parameter_count"),
        "transformer_matched_params": transformer.get("parameter_count"),
    }


def markdown_table(rows: List[Dict[str, object]]) -> str:
    def fmt(value: object) -> str:
        if value is None:
            return "NA"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return "NA"
        if math.isnan(number):
            return "NA"
        return f"{number:.4f}"

    lines = [
        "| B | Def-use pairs | Cross-block pairs | AST graph cross-block loss | Transformer matched cross-block loss | AST graph preservation | Transformer preservation |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['block_size']} | {float(row['definition_use_pairs']):.0f} | "
            f"{float(row['cross_block_pairs']):.0f} | {fmt(row['ast_graph_cross_block_use_loss'])} | "
            f"{fmt(row['transformer_matched_cross_block_use_loss'])} | "
            f"{fmt(row['ast_graph_cross_block_use_preservation'])} | "
            f"{fmt(row['transformer_matched_cross_block_use_preservation'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run block-size sensitivity diagnostics.")
    parser.add_argument("--block-sizes", type=parse_int_list, default=parse_int_list("32,64,128"))
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--ast-dim", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--case-set", choices=("tiny", "long", "all"), default="long")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, default=Path("results/block_size_sensitivity_32_64_128.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("results/block_size_sensitivity_32_64_128.md"))
    args = parser.parse_args()

    rows: List[Dict[str, object]] = []
    for block_size in args.block_sizes:
        probe_path = ROOT / "results" / f"structural_probe_b{block_size}.json"
        train_path = ROOT / "results" / f"diagnostic_poc_transformer_b{block_size}.json"
        work_dir = ROOT / "tmp" / f"diagnostic_block_b{block_size}"
        run_command(
            [
                sys.executable,
                "scripts/structural_probe_eval.py",
                "--output",
                str(probe_path),
                "--block-size",
                str(block_size),
                "--max-tokens",
                str(args.max_tokens),
                "--case-set",
                args.case_set,
            ]
        )
        run_command(
            [
                sys.executable,
                "scripts/diagnostic_poc_train.py",
                "--work-dir",
                str(work_dir),
                "--output",
                str(train_path),
                "--block-size",
                str(block_size),
                "--max-tokens",
                str(args.max_tokens),
                "--hidden-size",
                str(args.hidden_size),
                "--ast-dim",
                str(args.ast_dim),
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--case-set",
                args.case_set,
                "--variants",
                "transformer_matched,ast_graph",
            ]
        )
        rows.append(row_for(block_size, probe_path, train_path))

    result = {
        "purpose": "block-size sensitivity for tiny structural diagnostics; not a benchmark",
        "block_sizes": args.block_sizes,
        "steps": args.steps,
        "hidden_size": args.hidden_size,
        "ast_dim": args.ast_dim,
        "max_tokens": args.max_tokens,
        "case_set": args.case_set,
        "seed": args.seed,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown_table(rows), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown_output": str(args.markdown_output)}, indent=2))


if __name__ == "__main__":
    main()
