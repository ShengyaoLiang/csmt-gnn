"""
Small structural diagnostics for CSMT-GNN.

This script does not claim model quality. It creates tiny Python cases that
stress variable definitions, shadowing, and long-range use, then verifies that
the AST pipeline marks definition tokens and writes paired token/AST arrays that
can drive a minimal proof-of-concept training run.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary, tokenize_python_source


CASES: List[Tuple[str, str]] = [
    (
        "shadowing",
        """
value = 3
def scale(items):
    value = 10
    return [value * item for item in items]
result = scale([1, 2, 3])
""".strip(),
    ),
    (
        "long_range_import",
        """
import math

def area(radius):
    padding = 0
    for _ in range(8):
        padding += 1
    return math.pi * radius * radius
""".strip(),
    ),
    (
        "guarded_attribute",
        """
class Box:
    def __init__(self, value):
        self.value = value

def read(box):
    if box is not None:
        return box.value
    return 0
""".strip(),
    ),
]


def lexical_token_ids(source: str, max_tokens: int, vocab: Dict[str, int]) -> np.ndarray:
    spans = tokenize_python_source(source, max_tokens)
    ids = []
    for token, _, _ in spans:
        if token not in vocab:
            vocab[token] = len(vocab)
        ids.append(vocab[token])
    return np.asarray(ids, dtype=np.int64)


def run(output_dir: Path, block_size: int, max_tokens: int) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ast_dir = output_dir / "ast"
    token_dir = output_dir / "tokens"
    ast_dir.mkdir(exist_ok=True)
    token_dir.mkdir(exist_ok=True)

    vocab = TypeVocabulary()
    token_vocab = {"<PAD>": 0, "<UNK>": 1}
    extractor = ASTFeatureExtractor(ASTPreprocessConfig(block_size=block_size, max_tokens=max_tokens), vocab)
    summary = []
    for idx, (name, source) in enumerate(CASES):
        ast_ids, block_mask, token_mask = extractor.extract(source)
        tokens = lexical_token_ids(source, max_tokens, token_vocab)
        prefix = f"{idx}_{name}"
        np.save(token_dir / f"{prefix}_tokens.npy", tokens)
        np.save(ast_dir / f"{prefix}_ast_ids.npy", ast_ids)
        np.save(ast_dir / f"{prefix}_ast_mask.npy", block_mask)
        np.save(ast_dir / f"{prefix}_token_ast_mask.npy", token_mask)
        summary.append(
            {
                "name": name,
                "num_tokens": int(tokens.shape[0]),
                "num_blocks": int(ast_ids.shape[0]),
                "definition_tokens": int(token_mask.sum()),
                "definition_blocks": int(block_mask.sum()),
            }
        )

    (ast_dir / "ast_vocab.json").write_text(
        json.dumps(
            {
                "config": {"block_size": block_size, "max_tokens": max_tokens},
                "token_alignment": "python_lexical_tokens",
                "per_token_prefix": False,
                "type_vocab": vocab.to_json(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (token_dir / "token_vocab.json").write_text(json.dumps(token_vocab, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    if not all(item["definition_tokens"] > 0 for item in summary):
        raise SystemExit("At least one diagnostic case has no detected definition tokens.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create CSMT-GNN structural diagnostic arrays.")
    parser.add_argument("--output-dir", type=Path, default=Path("tmp/diagnostics"))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()
    run(args.output_dir, args.block_size, args.max_tokens)


if __name__ == "__main__":
    main()
