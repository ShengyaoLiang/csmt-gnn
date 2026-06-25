"""Validate paired token and AST NumPy artifacts before CSMT-GNN training.

This script is the intended preflight check before using
``train.py --no-input-range-validation``.  It checks the large token/AST id
arrays once, outside the hot training loop, while leaving dtype, shape, and
length validation active inside the model.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np


INTEGER_KINDS = {"i", "u"}
MASK_KINDS = {"b", "i", "u"}


@dataclass
class SampleReport:
    prefix: str
    token_length: int
    used_length: int
    required_blocks: int
    ast_blocks: int
    mask_width: int
    status: str
    issues: List[str]


def _ast_ids_path(ast_dir: Path, prefix: str) -> Path:
    direct = ast_dir / f"{prefix}_ast_ids.npy"
    legacy = ast_dir / f"{prefix}_ast_type_ids.npy"
    return direct if direct.exists() else legacy


def _ast_mask_path(ast_dir: Path, prefix: str) -> Path:
    token_level = ast_dir / f"{prefix}_token_ast_mask.npy"
    if token_level.exists():
        return token_level
    direct = ast_dir / f"{prefix}_ast_mask.npy"
    legacy = ast_dir / f"{prefix}_var_def_mask.npy"
    return direct if direct.exists() else legacy


def _load_num_ast_types(ast_dir: Path, fallback: Optional[int]) -> Optional[int]:
    if fallback is not None and fallback > 0:
        return fallback
    vocab_path = ast_dir / "ast_vocab.json"
    if not vocab_path.exists():
        return fallback
    data = json.loads(vocab_path.read_text(encoding="utf-8"))
    vocab = data.get("type_vocab", data)
    return len(vocab) if isinstance(vocab, dict) else fallback


def _check_integer_range(name: str, array: np.ndarray, upper_bound: int, issues: List[str]) -> None:
    if array.dtype.kind not in INTEGER_KINDS:
        issues.append(f"{name} must use an integer dtype, got {array.dtype}")
        return
    if array.size == 0:
        issues.append(f"{name} is empty")
        return
    min_id = int(array.min())
    max_id = int(array.max())
    if min_id < 0 or max_id >= upper_bound:
        issues.append(f"{name} ids must be in [0, {upper_bound - 1}], got min={min_id}, max={max_id}")


def _check_mask(name: str, array: np.ndarray, issues: List[str]) -> None:
    if array.dtype.kind not in MASK_KINDS:
        issues.append(f"{name} must use bool or integer dtype, got {array.dtype}")
    if array.ndim != 1:
        issues.append(f"{name} must be one-dimensional, got shape={tuple(array.shape)}")


def validate_sample(
    prefix: str,
    token_path: Path,
    ast_dir: Path,
    vocab_size: int,
    num_ast_types: Optional[int],
    block_size: int,
    max_tokens: int,
) -> SampleReport:
    issues: List[str] = []
    ast_path = _ast_ids_path(ast_dir, prefix)
    mask_path = _ast_mask_path(ast_dir, prefix)

    if not ast_path.exists():
        issues.append(f"missing AST id file: {ast_path.name}")
    if not mask_path.exists():
        issues.append(f"missing AST mask file: {mask_path.name}")

    tokens = np.load(token_path, mmap_mode="r")
    if tokens.ndim != 1:
        issues.append(f"tokens must be one-dimensional, got shape={tuple(tokens.shape)}")
    token_length = int(tokens.shape[0]) if tokens.ndim >= 1 else 0
    used_length = min(token_length, max_tokens)
    required_blocks = math.ceil(max(1, used_length) / block_size)
    if used_length < 2:
        issues.append("sample has fewer than two usable tokens for next-token training")
    _check_integer_range("tokens", tokens[:max_tokens], vocab_size, issues)

    ast_blocks = 0
    if ast_path.exists():
        ast_ids = np.load(ast_path, mmap_mode="r")
        if ast_ids.ndim != 2:
            issues.append(f"ast_ids must have shape [num_blocks, block_size], got shape={tuple(ast_ids.shape)}")
        else:
            ast_blocks = int(ast_ids.shape[0])
            if int(ast_ids.shape[1]) != block_size:
                issues.append(f"ast_ids second dimension must equal block_size={block_size}, got {ast_ids.shape[1]}")
            if ast_blocks < required_blocks:
                issues.append(f"ast_ids has {ast_blocks} blocks but {required_blocks} are required for used_length={used_length}")
        if num_ast_types is not None:
            _check_integer_range("ast_ids", ast_ids, num_ast_types, issues)
        elif ast_ids.dtype.kind not in INTEGER_KINDS:
            issues.append(f"ast_ids must use an integer dtype, got {ast_ids.dtype}")

    mask_width = 0
    if mask_path.exists():
        mask = np.load(mask_path, mmap_mode="r")
        mask_width = int(mask.size)
        _check_mask("ast_mask", mask, issues)
        if mask.ndim == 1 and mask_width not in {token_length, used_length, ast_blocks, required_blocks}:
            issues.append(
                "ast_mask width should match token length, used length, AST blocks, or required blocks; "
                f"got width={mask_width}"
            )

    return SampleReport(
        prefix=prefix,
        token_length=token_length,
        used_length=used_length,
        required_blocks=required_blocks,
        ast_blocks=ast_blocks,
        mask_width=mask_width,
        status="ok" if not issues else "error",
        issues=issues,
    )


def iter_token_files(data_dir: Path) -> Iterable[Tuple[str, Path]]:
    for path in sorted(data_dir.glob("*_tokens.npy")):
        yield path.name[: -len("_tokens.npy")], path


def run(args: argparse.Namespace) -> Dict[str, object]:
    if args.vocab_size <= 0:
        raise ValueError("--vocab-size must be positive")
    if args.block_size <= 0:
        raise ValueError("--block-size must be positive")
    if args.max_tokens <= 1:
        raise ValueError("--max-tokens must be at least 2")
    num_ast_types = _load_num_ast_types(args.ast_path, args.num_ast_types)
    if num_ast_types is not None and num_ast_types <= 0:
        raise ValueError("--num-ast-types must be positive when provided")

    reports: List[SampleReport] = []
    for index, (prefix, token_path) in enumerate(iter_token_files(args.data_path)):
        if args.max_samples is not None and index >= args.max_samples:
            break
        reports.append(
            validate_sample(
            prefix=prefix,
            token_path=token_path,
            ast_dir=args.ast_path,
            vocab_size=args.vocab_size,
            num_ast_types=num_ast_types,
            block_size=args.block_size,
            max_tokens=args.max_tokens,
        )
        )
    errors = [report for report in reports if report.issues]
    result = {
        "purpose": "CSMT-GNN data pipeline preflight; run before --no-input-range-validation",
        "data_path": str(args.data_path),
        "ast_path": str(args.ast_path),
        "vocab_size": args.vocab_size,
        "num_ast_types": num_ast_types,
        "block_size": args.block_size,
        "max_tokens": args.max_tokens,
        "samples_checked": len(reports),
        "samples_with_errors": len(errors),
        "ok": len(reports) > 0 and not errors,
        "reports": [asdict(report) for report in reports],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate token/AST NumPy artifacts before CSMT-GNN training.")
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--ast-path", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--num-ast-types", type=int)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    result = run(parse_args())
    print(json.dumps(result, indent=2))
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
