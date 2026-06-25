"""
Measure prefix-AST degradation for a finished Python source file.

This is a Stage-0 diagnostic for the paper. It compares AST ids produced from a
complete-file parse with ids produced from generation-style prefixes. The output
is a JSON record with fallback rate, unknown rate, prefix/full disagreement, and
latency. It is not a model-quality benchmark.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary
from inference_ast import load_type_vocab


def _valid_ids(ast_ids: np.ndarray, pad_id: int) -> np.ndarray:
    flat = ast_ids.reshape(-1)
    return flat[flat != pad_id]


def _load_vocab(path: Optional[Path]) -> TypeVocabulary:
    if path is None:
        return TypeVocabulary()
    return load_type_vocab(path)


def _source_prefix_from_byte(source_bytes: bytes, end_byte: int) -> str:
    return source_bytes[:end_byte].decode("utf-8", errors="ignore")


def measure(
    source: str,
    block_size: int,
    max_tokens: int,
    vocab_path: Optional[Path],
    tokenizer_name_or_path: Optional[str],
    repeat: int,
) -> Dict[str, object]:
    repeat = max(1, repeat)
    vocab = _load_vocab(vocab_path)
    pad_id = vocab.type_to_id["<PAD>"]
    unknown_id = vocab.type_to_id["<UNKNOWN>"]

    full_extractor = ASTFeatureExtractor(
        ASTPreprocessConfig(block_size=block_size, max_tokens=max_tokens, prefix_parse=False),
        vocab,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )
    full_ids, _, _ = full_extractor.extract(source)
    token_spans = full_extractor.tokenize(source)

    prefix_extractor = ASTFeatureExtractor(
        ASTPreprocessConfig(block_size=block_size, max_tokens=max_tokens, prefix_parse=True),
        vocab,
        tokenizer_name_or_path=tokenizer_name_or_path,
    )

    source_bytes = source.encode("utf-8")
    prefix_token_ids = []
    fallback_tokens = 0
    start = time.perf_counter()
    for _ in range(repeat):
        prefix_token_ids = []
        fallback_tokens = 0
        for tok_idx, (_, _, end_byte) in enumerate(token_spans):
            prefix_source = _source_prefix_from_byte(source_bytes, end_byte)
            prefix_ids, _, _ = prefix_extractor.extract(prefix_source)
            if prefix_ids.size == 0:
                prefix_token_ids.append(unknown_id)
                fallback_tokens += 1
                continue
            flat = prefix_ids.reshape(-1)
            prefix_token_ids.append(int(flat[min(tok_idx, flat.size - 1)]))
            fallback_mask = prefix_extractor.last_stats.get("fallback_token_mask", [])
            if tok_idx < len(fallback_mask):
                fallback_tokens += int(bool(fallback_mask[tok_idx]))
            elif not bool(prefix_extractor.last_stats.get("tree_sitter_available", True)):
                fallback_tokens += 1
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    full_valid = _valid_ids(full_ids, pad_id)
    prefix_valid = np.asarray(prefix_token_ids, dtype=np.int32)
    compare_len = min(full_valid.size, prefix_valid.size)
    if compare_len:
        prefix_full_divergence = float(np.mean(full_valid[:compare_len] != prefix_valid[:compare_len]))
    else:
        prefix_full_divergence = 0.0

    unknown_tokens = int(np.sum(prefix_valid[:compare_len] == unknown_id)) if compare_len else 0
    token_count = int(max(1, len(token_spans)))

    return {
        "purpose": "prefix/full AST degradation measurement; not a model benchmark",
        "num_tokens": int(len(token_spans)),
        "block_size": int(block_size),
        "max_tokens": int(max_tokens),
        "vocab_frozen": vocab_path is not None,
        "tree_sitter_available": bool(full_extractor.last_stats.get("tree_sitter_available", False)),
        "fallback_tokens": fallback_tokens,
        "fallback_rate": float(fallback_tokens / token_count),
        "unknown_tokens": unknown_tokens,
        "unknown_rate": float(unknown_tokens / token_count),
        "prefix_full_divergence": prefix_full_divergence,
        "full_ast_tokens": int(full_valid.size),
        "prefix_ast_tokens": int(prefix_valid.size),
        "repeat": repeat,
        "avg_ms": float(elapsed_ms / repeat),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure prefix/full AST feature degradation.")
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--tokenizer-name-or-path", type=str)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--encoding", type=str, default="utf-8")
    args = parser.parse_args()

    source = args.source_file.read_text(encoding=args.encoding, errors="replace")
    result = measure(
        source=source,
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        vocab_path=args.vocab_path,
        tokenizer_name_or_path=args.tokenizer_name_or_path,
        repeat=args.repeat,
    )
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
