"""Generate simple CSMT-vs-Transformer structural cost tables.

The script counts causal attention edges, not wall-clock time.  It is a small
mathematical audit companion for the paper: dense token attention uses
L(L+1)/2 causal token edges, while CSMT-style communication uses causal
block-local token edges plus a causal block graph.  The table is meant to make
the block-size trade-off explicit before any performance claim is made.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List


@dataclass(frozen=True)
class CostRow:
    sequence_length: int
    block_size: int
    num_blocks: int
    dense_causal_edges: int
    block_local_edges: int
    block_graph_edges: int
    csmt_edges: int
    edge_ratio_vs_dense: float
    asymptotic_proxy: float
    continuous_optimal_block: float


def causal_edges(length: int) -> int:
    if length <= 0:
        raise ValueError(f"length must be positive, got {length}")
    return length * (length + 1) // 2


def block_local_edges(sequence_length: int, block_size: int) -> int:
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    full_blocks, tail = divmod(sequence_length, block_size)
    total = full_blocks * causal_edges(block_size)
    if tail:
        total += causal_edges(tail)
    return total


def cost_row(sequence_length: int, block_size: int, a: float = 1.0, b: float = 1.0) -> CostRow:
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if block_size <= 0:
        raise ValueError(f"block_size must be positive, got {block_size}")
    if a <= 0 or b <= 0:
        raise ValueError("a and b must be positive")

    num_blocks = math.ceil(sequence_length / block_size)
    dense = causal_edges(sequence_length)
    local = block_local_edges(sequence_length, block_size)
    graph = causal_edges(num_blocks)
    csmt = local + graph
    proxy = a * sequence_length * block_size + b * (sequence_length**2) / (block_size**2)
    optimum = ((2.0 * b * sequence_length) / a) ** (1.0 / 3.0)
    return CostRow(
        sequence_length=sequence_length,
        block_size=block_size,
        num_blocks=num_blocks,
        dense_causal_edges=dense,
        block_local_edges=local,
        block_graph_edges=graph,
        csmt_edges=csmt,
        edge_ratio_vs_dense=csmt / dense,
        asymptotic_proxy=proxy,
        continuous_optimal_block=optimum,
    )


def parse_int_list(text: str) -> List[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a CSMT-vs-Transformer structural cost table.")
    parser.add_argument("--lengths", type=parse_int_list, default=parse_int_list("512,1024,2048,4096"))
    parser.add_argument("--block-sizes", type=parse_int_list, default=parse_int_list("16,32,64,128"))
    parser.add_argument("--a", type=float, default=1.0, help="Coefficient for local token work in aLB+bL^2/B^2.")
    parser.add_argument("--b", type=float, default=1.0, help="Coefficient for graph work in aLB+bL^2/B^2.")
    parser.add_argument("--output", type=Path, default=Path("results/architecture_cost_table.json"))
    args = parser.parse_args()

    rows = [
        cost_row(sequence_length, block_size, a=args.a, b=args.b)
        for sequence_length in args.lengths
        for block_size in args.block_sizes
    ]
    result = {
        "purpose": "structural edge-count and block-size trade-off audit; not a runtime benchmark",
        "cost_model": "dense causal edges vs block-local causal token edges plus causal block-graph edges",
        "proxy": "a*L*B + b*L^2/B^2",
        "a": args.a,
        "b": args.b,
        "rows": [asdict(row) for row in rows],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
