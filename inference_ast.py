"""
Incremental AST features for autoregressive CSMT-GNN inference.

The training preprocessor can parse complete files offline. During generation,
the prefix is often syntactically incomplete, so this module builds AST features
from the current prefix and falls back to lexical ids when parsing cannot
recover a useful node. It keeps the inference contract explicit: AST conditioning
is available during generation, but it is approximate and must be measured.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary


@dataclass(frozen=True)
class IncrementalASTConfig:
    block_size: int = 64
    max_tokens: int = 2048
    tokenizer_name_or_path: Optional[str] = None
    vocab_path: Optional[Path] = None


@dataclass(frozen=True)
class IncrementalASTResult:
    ast_ids: np.ndarray
    token_var_mask: np.ndarray
    fallback_tokens: int
    fallback_rate: float


def load_type_vocab(path: Optional[Path]) -> TypeVocabulary:
    if path is None or not path.exists():
        return TypeVocabulary()
    data = json.loads(path.read_text(encoding="utf-8"))
    vocab = data.get("type_vocab", data)
    return TypeVocabulary(vocab, frozen=True)


class IncrementalASTBuilder:
    """Build model-ready AST arrays from an incomplete generation prefix."""

    def __init__(self, config: IncrementalASTConfig) -> None:
        preprocess_config = ASTPreprocessConfig(
            block_size=config.block_size,
            max_tokens=config.max_tokens,
            prefix_parse=True,
        )
        self.config = config
        self.vocab = load_type_vocab(config.vocab_path)
        self.extractor = ASTFeatureExtractor(
            preprocess_config,
            self.vocab,
            tokenizer_name_or_path=config.tokenizer_name_or_path,
        )

    def build(self, prefix_source: str) -> IncrementalASTResult:
        ast_ids, _, token_var_mask = self.extractor.extract(prefix_source)
        stats = self.extractor.last_stats
        return IncrementalASTResult(
            ast_ids=ast_ids,
            token_var_mask=token_var_mask,
            fallback_tokens=int(stats.get("fallback_tokens", 0)),
            fallback_rate=float(stats.get("fallback_rate", 0.0)),
        )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build incremental AST features for a source prefix.")
    parser.add_argument("--source-file", type=Path, required=True)
    parser.add_argument("--vocab-path", type=Path)
    parser.add_argument("--tokenizer-name-or-path", type=str)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--repeat", type=int, default=1, help="Repeat feature building to estimate average latency.")
    args = parser.parse_args()

    builder = IncrementalASTBuilder(
        IncrementalASTConfig(
            block_size=args.block_size,
            max_tokens=args.max_tokens,
            tokenizer_name_or_path=args.tokenizer_name_or_path,
            vocab_path=args.vocab_path,
        )
    )
    source = args.source_file.read_text(encoding="utf-8", errors="replace")
    repeat = max(1, args.repeat)
    start = time.perf_counter()
    result = None
    for _ in range(repeat):
        result = builder.build(source)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    assert result is not None
    ast_ids = result.ast_ids
    token_mask = result.token_var_mask
    valid_ast_ids = ast_ids[ast_ids != builder.vocab.type_to_id["<PAD>"]]
    unknown_tokens = int((valid_ast_ids == builder.vocab.type_to_id["<UNKNOWN>"]).sum())
    total_ast_tokens = int(valid_ast_ids.size)
    unknown_rate = unknown_tokens / max(1, total_ast_tokens)
    print(
        " ".join(
            [
                f"ast_ids_shape={ast_ids.shape}",
                f"token_var_mask_shape={token_mask.shape}",
                f"token_defs={int(token_mask.sum())}",
                f"fallback_tokens={result.fallback_tokens}",
                f"fallback_rate={result.fallback_rate:.4f}",
                f"unknown_tokens={unknown_tokens}",
                f"unknown_rate={unknown_rate:.4f}",
                f"vocab_size={len(builder.vocab.type_to_id)}",
                f"repeat={repeat}",
                f"avg_ms={elapsed_ms / repeat:.3f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
