from __future__ import annotations

import unittest

from ast_preprocessor import ASTFeatureExtractor, ASTPreprocessConfig, TypeVocabulary


def definition_tokens(source: str) -> set[str]:
    extractor = ASTFeatureExtractor(ASTPreprocessConfig(block_size=4, max_tokens=128), TypeVocabulary())
    _, _, token_mask = extractor.extract(source)
    spans = extractor.tokenize(source)
    return {token for (token, _, _), is_definition in zip(spans, token_mask) if bool(is_definition)}


class ASTDefinitionMaskTests(unittest.TestCase):
    def test_assignment_and_parameter_definitions(self) -> None:
        defs = definition_tokens("def f(x, y):\n    value = x + y\n    return value\n")
        self.assertIn("f", defs)
        self.assertIn("x", defs)
        self.assertIn("y", defs)
        self.assertIn("value", defs)

    def test_import_aliases_are_definitions(self) -> None:
        defs = definition_tokens("import math as m\nfrom os import path as p\narea = m.pi\n")
        self.assertIn("m", defs)
        self.assertIn("p", defs)
        self.assertIn("area", defs)
        self.assertNotIn("os", defs)
        self.assertNotIn("path", defs)

    def test_attribute_assignment_marks_attribute_name(self) -> None:
        defs = definition_tokens("class Box:\n    def __init__(self, value):\n        self.value = value\n")
        self.assertIn("Box", defs)
        self.assertIn("__init__", defs)
        self.assertIn("self", defs)
        self.assertIn("value", defs)

    def test_destructuring_and_for_targets(self) -> None:
        defs = definition_tokens("a, (b, c) = item\nfor key, value in pairs:\n    total = value\n")
        for expected in {"a", "b", "c", "key", "value", "total"}:
            self.assertIn(expected, defs)

    def test_use_only_code_is_not_all_definitions(self) -> None:
        source = "result = value + other\nprint(result)\n"
        extractor = ASTFeatureExtractor(ASTPreprocessConfig(block_size=4, max_tokens=128), TypeVocabulary())
        _, _, token_mask = extractor.extract(source)
        spans = extractor.tokenize(source)
        marked = [token for (token, _, _), is_definition in zip(spans, token_mask) if bool(is_definition)]
        self.assertIn("result", marked)
        self.assertNotIn("value", marked)
        self.assertNotIn("other", marked)
        self.assertLess(sum(bool(x) for x in token_mask), len(spans))


if __name__ == "__main__":
    unittest.main()
