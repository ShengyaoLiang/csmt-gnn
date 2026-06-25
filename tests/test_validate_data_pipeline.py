from __future__ import annotations

import argparse
import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.validate_data_pipeline import run


def make_args(root: Path, **overrides):
    data = {
        "data_path": root / "tokens",
        "ast_path": root / "ast",
        "vocab_size": 16,
        "num_ast_types": 8,
        "block_size": 4,
        "max_tokens": 16,
        "max_samples": None,
        "output": None,
    }
    data.update(overrides)
    return argparse.Namespace(**data)


class ValidateDataPipelineTests(unittest.TestCase):
    def write_sample(
        self,
        root: Path,
        prefix: str = "0",
        tokens=None,
        ast_ids=None,
        ast_mask=None,
    ) -> None:
        token_dir = root / "tokens"
        ast_dir = root / "ast"
        token_dir.mkdir(parents=True, exist_ok=True)
        ast_dir.mkdir(parents=True, exist_ok=True)
        if tokens is None:
            tokens = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        if ast_ids is None:
            ast_ids = np.zeros((2, 4), dtype=np.int64)
        if ast_mask is None:
            ast_mask = np.array([True, False, True, False, False], dtype=np.bool_)
        np.save(token_dir / f"{prefix}_tokens.npy", tokens)
        np.save(ast_dir / f"{prefix}_ast_ids.npy", ast_ids)
        np.save(ast_dir / f"{prefix}_token_ast_mask.npy", ast_mask)

    def test_valid_sample_passes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_sample(root)
            result = run(make_args(root))
            self.assertTrue(result["ok"])
            self.assertEqual(result["samples_checked"], 1)
            self.assertEqual(result["samples_with_errors"], 0)

    def test_invalid_token_and_ast_ranges_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_sample(
                root,
                tokens=np.array([1, 2, 16], dtype=np.int64),
                ast_ids=np.array([[0, 1, 2, 9]], dtype=np.int64),
            )
            result = run(make_args(root))
            self.assertFalse(result["ok"])
            issues = " ".join(result["reports"][0]["issues"])
            self.assertIn("tokens ids must be", issues)
            self.assertIn("ast_ids ids must be", issues)

    def test_shape_and_mask_errors_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            self.write_sample(
                root,
                ast_ids=np.zeros((1, 3), dtype=np.int64),
                ast_mask=np.zeros((1, 4), dtype=np.bool_),
            )
            result = run(make_args(root))
            self.assertFalse(result["ok"])
            issues = " ".join(result["reports"][0]["issues"])
            self.assertIn("second dimension must equal block_size", issues)
            self.assertIn("ast_mask must be one-dimensional", issues)


if __name__ == "__main__":
    unittest.main()
