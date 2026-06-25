"""
Evaluate structural probes in the tiny diagnostic corpus.

This script measures whether diagnostic cases contain nontrivial definition-use
structure: definition tokens, use tokens, token distance, and cross-block
distance. It does not evaluate model accuracy.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary
from diagnostics import CASES


def token_records(source: str, block_size: int, max_tokens: int) -> List[Dict[str, object]]:
    extractor = ASTFeatureExtractor(ASTPreprocessConfig(block_size=block_size, max_tokens=max_tokens), TypeVocabulary())
    _, _, token_mask = extractor.extract(source)
    spans = extractor.tokenize(source)
    records = []
    for idx, ((token, _, _), is_def) in enumerate(zip(spans, token_mask)):
        records.append({"token": token, "index": idx, "block": idx // block_size, "is_definition": bool(is_def)})
    return records


def first_definition_use(records: List[Dict[str, object]]) -> List[Dict[str, object]]:
    first_defs: Dict[str, Dict[str, object]] = {}
    pairs: List[Dict[str, object]] = []
    for rec in records:
        token = str(rec["token"])
        if not token.isidentifier():
            continue
        if bool(rec["is_definition"]):
            first_defs[token] = rec
            continue
        definition = first_defs.get(token)
        if definition is None:
            continue
        distance = int(rec["index"]) - int(definition["index"])
        block_distance = int(rec["block"]) - int(definition["block"])
        if distance > 0:
            pairs.append(
                {
                    "token": token,
                    "definition_index": int(definition["index"]),
                    "use_index": int(rec["index"]),
                    "distance": distance,
                    "definition_block": int(definition["block"]),
                    "use_block": int(rec["block"]),
                    "block_distance": block_distance,
                    "cross_block": block_distance > 0,
                }
            )
    return pairs


def summarize_case(name: str, source: str, block_size: int, max_tokens: int) -> Dict[str, object]:
    records = token_records(source, block_size, max_tokens)
    pairs = first_definition_use(records)
    definition_tokens = [rec for rec in records if rec["is_definition"]]
    cross_block_pairs = [pair for pair in pairs if pair["cross_block"]]
    return {
        "name": name,
        "num_tokens": len(records),
        "num_blocks": (len(records) + block_size - 1) // block_size,
        "definition_tokens": len(definition_tokens),
        "definition_blocks": len({rec["block"] for rec in definition_tokens}),
        "definition_use_pairs": len(pairs),
        "cross_block_pairs": len(cross_block_pairs),
        "max_token_distance": max((pair["distance"] for pair in pairs), default=0),
        "max_block_distance": max((pair["block_distance"] for pair in pairs), default=0),
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate tiny structural diagnostic probes.")
    parser.add_argument("--output", type=Path, default=Path("results/structural_probe_eval.json"))
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    args = parser.parse_args()

    result = {
        "purpose": "structural diagnostic coverage; not model accuracy",
        "block_size": args.block_size,
        "cases": [summarize_case(name, source, args.block_size, args.max_tokens) for name, source in CASES],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
