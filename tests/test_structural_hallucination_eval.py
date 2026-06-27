from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from scripts.diagnostic_poc_train import load_num_ast_types
from scripts.block_size_sensitivity import parse_int_list
from scripts.structural_hallucination_eval import CASES, evaluate_case, summarize


class StructuralHallucinationEvalTests(unittest.TestCase):
    def test_reference_cases_preserve_required_dependencies(self) -> None:
        results = [evaluate_case(case) for case in CASES]
        summary = summarize(results)
        self.assertEqual(summary["dependency_preservation_rate"], 1.0)
        self.assertEqual(summary["parse_success_rate"], 1.0)

    def test_missing_import_and_guard_reduce_preservation(self) -> None:
        long_range = next(case for case in CASES if case.name == "long_range_import")
        no_import = evaluate_case(long_range, "def area(radius):\n    return pi * radius * radius\n")
        self.assertIn("missing_import:math", no_import.issues)
        self.assertLess(no_import.dependency_preservation_rate, 1.0)

        guarded = next(case for case in CASES if case.name == "guarded_attribute")
        no_guard = evaluate_case(guarded, "def read(box):\n    return box.value\n")
        self.assertIn("missing_guarded_attribute:box.value", no_guard.issues)
        self.assertLess(no_guard.dependency_preservation_rate, 1.0)

    def test_syntax_error_fails_all_dependencies(self) -> None:
        case = CASES[0]
        result = evaluate_case(case, "def broken(:\n    pass")
        self.assertFalse(result.parse_ok)
        self.assertEqual(result.dependency_preserved, 0)
        self.assertEqual(result.dependency_preservation_rate, 0.0)

    def test_parse_int_list_requires_values(self) -> None:
        self.assertEqual(parse_int_list("32,64,128"), [32, 64, 128])
        with self.assertRaises(Exception):
            parse_int_list("")

    def test_ast_type_count_uses_max_id_not_vocab_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ast_dir = Path(temp_dir) / "ast"
            ast_dir.mkdir()
            (ast_dir / "ast_vocab.json").write_text(
                json.dumps({"type_vocab": {"<PAD>": 0, "<UNKNOWN>": 1, "identifier": 18}}),
                encoding="utf-8",
            )
            self.assertEqual(load_num_ast_types(Path(temp_dir)), 19)

    def test_ast_type_count_scans_arrays_when_metadata_underestimates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            ast_dir = Path(temp_dir) / "ast"
            ast_dir.mkdir()
            (ast_dir / "ast_vocab.json").write_text(
                json.dumps({"type_vocab": {"<PAD>": 0, "<UNKNOWN>": 1, "identifier": 2}}),
                encoding="utf-8",
            )
            np.save(ast_dir / "0_case_ast_ids.npy", np.array([[0, 2, 23]], dtype=np.int64))
            self.assertEqual(load_num_ast_types(Path(temp_dir)), 24)


if __name__ == "__main__":
    unittest.main()
