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
from typing import Dict, List, Sequence, Tuple

import numpy as np

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary, tokenize_python_source


TINY_CASES: List[Tuple[str, str]] = [
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


def _long_filler(indent: str, count: int) -> str:
    return "\n".join(f"{indent}pad_{idx} = {idx}" for idx in range(count))


LONG_CASES: List[Tuple[str, str]] = [
    (
        "long_scope_b128",
        f"""
def carry(anchor):
    local_value = anchor
{_long_filler("    ", 52)}
    return local_value + anchor
result = carry(3)
""".strip(),
    ),
    (
        "long_import_b128",
        f"""
import math

def area(radius):
{_long_filler("    ", 52)}
    return math.pi * radius * radius
""".strip(),
    ),
    (
        "long_guard_b128",
        f"""
def read(box):
{_long_filler("    ", 52)}
    if box is not None:
        return box.value
    return 0
""".strip(),
    ),
]


CASES: List[Tuple[str, str]] = TINY_CASES


def select_cases(case_set: str) -> Sequence[Tuple[str, str]]:
    if case_set == "tiny":
        return TINY_CASES
    if case_set == "long":
        return LONG_CASES
    if case_set == "all":
        return [*TINY_CASES, *LONG_CASES]
    raise ValueError(f"unknown case_set={case_set!r}; expected tiny, long, or all")


def lexical_token_ids(source: str, max_tokens: int, vocab: Dict[str, int]) -> np.ndarray:
    spans = tokenize_python_source(source, max_tokens)
    ids = []
    for token, _, _ in spans:
        if token not in vocab:
            vocab[token] = len(vocab)
        ids.append(vocab[token])
    return np.asarray(ids, dtype=np.int64)


def run(output_dir: Path, block_size: int, max_tokens: int, case_set: str = "tiny") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ast_dir = output_dir / "ast"
    token_dir = output_dir / "tokens"
    ast_dir.mkdir(exist_ok=True)
    token_dir.mkdir(exist_ok=True)

    vocab = TypeVocabulary()
    token_vocab = {"<PAD>": 0, "<UNK>": 1}
    extractor = ASTFeatureExtractor(ASTPreprocessConfig(block_size=block_size, max_tokens=max_tokens), vocab)
    summary = []
    cases = select_cases(case_set)
    for idx, (name, source) in enumerate(cases):
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
                "case_set": case_set,
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
    parser.add_argument("--case-set", choices=("tiny", "long", "all"), default="tiny")
    args = parser.parse_args()
    run(args.output_dir, args.block_size, args.max_tokens, case_set=args.case_set)


if __name__ == "__main__":
    main()
