"""
Offline AST feature builder for CSMT-GNN.

The preprocessor writes integer AST type ids and block-level variable-definition
masks. Training keeps the AST embedding table inside the model, so AST
representations are trainable and the data pipeline carries compact integers.

Examples:
    python ast_preprocessor.py --source-file example.py --output-dir ast_data
    python ast_preprocessor.py --source-dir corpus --glob "**/*.py" --output-dir ast_data
"""

from __future__ import annotations

import argparse
import ast
import io
import json
import os
import tokenize
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser

    _PY_LANGUAGE = Language(tspython.language())
    _PARSER = Parser()
    if hasattr(_PARSER, "set_language"):
        _PARSER.set_language(_PY_LANGUAGE)
    else:
        _PARSER.language = _PY_LANGUAGE
except Exception:
    _PY_LANGUAGE = None
    _PARSER = None


TokenSpan = Tuple[str, int, int]


@dataclass(frozen=True)
class ASTPreprocessConfig:
    block_size: int = 64
    max_tokens: int = 2048
    prefix_parse: bool = False
    pad_id: int = 0
    unknown_id: int = 1


class TypeVocabulary:
    """Stable AST type vocabulary with reserved PAD and UNKNOWN ids.

    Training-time preprocessing may grow the vocabulary. Inference-time feature
    building should usually freeze it so unseen node types map to UNKNOWN
    instead of creating ids that the model embedding table never learned.
    """

    def __init__(self, initial: Optional[Dict[str, int]] = None, frozen: bool = False) -> None:
        if initial is None:
            self.type_to_id: Dict[str, int] = {"<PAD>": 0, "<UNKNOWN>": 1}
        else:
            self.type_to_id = dict(initial)
            self.type_to_id.setdefault("<PAD>", 0)
            self.type_to_id.setdefault("<UNKNOWN>", 1)
        self.frozen = frozen

    def id_for(self, node_type: str) -> int:
        if not node_type:
            return self.type_to_id["<UNKNOWN>"]
        if node_type not in self.type_to_id:
            if self.frozen:
                return self.type_to_id["<UNKNOWN>"]
            self.type_to_id[node_type] = len(self.type_to_id)
        return self.type_to_id[node_type]

    def to_json(self) -> Dict[str, int]:
        return dict(sorted(self.type_to_id.items(), key=lambda item: item[1]))


PY_VAR_DEF_NODE_TYPES = {
    "assignment",
    "augmented_assignment",
    "for_statement",
    "with_statement",
    "function_definition",
    "class_definition",
    "typed_parameter",
    "default_parameter",
    "keyword_argument",
    "pattern_list",
}

PY_VAR_DEF_PARENT_TYPES = {
    "assignment",
    "augmented_assignment",
    "for_statement",
    "with_statement",
    "function_definition",
    "class_definition",
    "typed_parameter",
    "default_parameter",
    "pattern_list",
}


def _line_start_byte_offsets(source: str) -> List[int]:
    starts = [0]
    offset = 0
    for line in source.splitlines(keepends=True):
        offset += len(line.encode("utf-8"))
        starts.append(offset)
    return starts


def _point_to_byte(line_starts: Sequence[int], source_lines: Sequence[str], point: Tuple[int, int]) -> int:
    line_no, col = point
    if line_no >= len(source_lines):
        return line_starts[-1]
    prefix = source_lines[line_no][:col]
    return line_starts[line_no] + len(prefix.encode("utf-8"))


def tokenize_python_source(source: str, max_tokens: int) -> List[TokenSpan]:
    """Tokenize Python source and return (token, start_byte, end_byte)."""

    line_starts = _line_start_byte_offsets(source)
    source_lines = source.splitlines(keepends=True) or [source]
    spans: List[TokenSpan] = []

    try:
        reader = io.BytesIO(source.encode("utf-8")).readline
        for tok in tokenize.tokenize(reader):
            if tok.type in {
                tokenize.ENCODING,
                tokenize.ENDMARKER,
                tokenize.NEWLINE,
                tokenize.NL,
                tokenize.INDENT,
                tokenize.DEDENT,
            }:
                continue
            start = _point_to_byte(line_starts, source_lines, (tok.start[0] - 1, tok.start[1]))
            end = _point_to_byte(line_starts, source_lines, (tok.end[0] - 1, tok.end[1]))
            spans.append((tok.string, start, end))
            if len(spans) >= max_tokens:
                break
    except tokenize.TokenError:
        spans = _whitespace_tokenize(source, max_tokens)

    return spans


def load_hf_tokenizer(tokenizer_name_or_path: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install transformers to use --tokenizer-name-or-path.") from exc
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise RuntimeError("--tokenizer-name-or-path must resolve to a HuggingFace fast tokenizer.")
    return tokenizer


def tokenize_with_hf_offsets(source: str, tokenizer, max_tokens: int) -> List[TokenSpan]:
    """Tokenize with a HuggingFace fast tokenizer and return UTF-8 byte spans.

    This is the preferred mode when AST artifacts will be paired with model token
    ids. The returned spans follow the same tokenization as the training data,
    avoiding the lexical-token/BPE mismatch.
    """

    encoded = tokenizer(
        source,
        return_offsets_mapping=True,
        add_special_tokens=False,
        truncation=True,
        max_length=max_tokens,
    )
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"])
    spans: List[TokenSpan] = []
    for token, (char_start, char_end) in zip(tokens, encoded["offset_mapping"]):
        if char_end <= char_start:
            continue
        start = len(source[:char_start].encode("utf-8"))
        end = len(source[:char_end].encode("utf-8"))
        spans.append((token, start, end))
    return spans[:max_tokens]


def _whitespace_tokenize(source: str, max_tokens: int) -> List[TokenSpan]:
    spans: List[TokenSpan] = []
    cursor = 0
    encoded = source.encode("utf-8")
    for word in source.split():
        char_start = source.find(word, cursor)
        if char_start < 0:
            continue
        char_end = char_start + len(word)
        start = len(source[:char_start].encode("utf-8"))
        end = len(source[:char_end].encode("utf-8"))
        if start <= len(encoded):
            spans.append((word, start, end))
        cursor = char_end
        if len(spans) >= max_tokens:
            break
    return spans


def _ancestor_has_type(node, target_types: Iterable[str], max_depth: int = 10) -> bool:
    target = set(target_types)
    cur = node
    depth = 0
    while cur is not None and depth <= max_depth:
        if getattr(cur, "type", None) in target:
            return True
        cur = getattr(cur, "parent", None)
        depth += 1
    return False


def _node_or_ancestor_is_error(node, max_depth: int = 10) -> bool:
    cur = node
    depth = 0
    while cur is not None and depth <= max_depth:
        if getattr(cur, "type", None) == "ERROR" or bool(getattr(cur, "is_missing", False)):
            return True
        cur = getattr(cur, "parent", None)
        depth += 1
    return False


def _same_node_span(left, right) -> bool:
    return (
        getattr(left, "type", None) == getattr(right, "type", None)
        and getattr(left, "start_byte", None) == getattr(right, "start_byte", None)
        and getattr(left, "end_byte", None) == getattr(right, "end_byte", None)
    )


ASSIGNMENT_OPERATORS = {"=", "+=", "-=", "*=", "/=", "%=", "**=", "//=", "@=", "&=", "|=", "^=", ">>=", "<<="}


def _iter_nodes(node):
    stack = [node] if node is not None else []
    while stack:
        cur = stack.pop()
        yield cur
        stack.extend(reversed(list(getattr(cur, "children", []))))


def _node_contains(ancestor, child) -> bool:
    return (
        getattr(ancestor, "start_byte", -1) <= getattr(child, "start_byte", -2)
        and getattr(child, "end_byte", -2) <= getattr(ancestor, "end_byte", -1)
    )


def _previous_named_sibling_type(node) -> Optional[str]:
    parent = getattr(node, "parent", None)
    if parent is None:
        return None
    previous = None
    for child in getattr(parent, "children", []):
        if _same_node_span(child, node):
            return getattr(previous, "type", None)
        if getattr(child, "is_named", True):
            previous = child
    return None


def _child_by_field(node, field_name: str):
    if node is None or not hasattr(node, "child_by_field_name"):
        return None
    try:
        return node.child_by_field_name(field_name)
    except Exception:
        return None


def _is_assignment_lhs(node, assignment) -> bool:
    for child in getattr(assignment, "children", []):
        child_type = getattr(child, "type", None)
        if child_type in ASSIGNMENT_OPERATORS:
            return False
        if _node_contains(child, node):
            return True
    return False


def _is_for_target(node, for_statement) -> bool:
    for child in getattr(for_statement, "children", []):
        child_type = getattr(child, "type", None)
        if child_type == "in":
            return False
        if _node_contains(child, node):
            return True
    return False


def _is_with_alias(node, with_statement) -> bool:
    previous = None
    for child in getattr(with_statement, "children", []):
        child_type = getattr(child, "type", None)
        if _node_contains(child, node):
            return previous == "as"
        previous = child_type
    return False


def _is_import_alias(node) -> bool:
    parent = getattr(node, "parent", None)
    parent_type = getattr(parent, "type", None)
    if parent_type == "aliased_import":
        alias = _child_by_field(parent, "alias")
        if alias is not None:
            return _node_contains(alias, node)
        previous_type = _previous_named_sibling_type(node)
        return previous_type == "as"
    if parent_type == "dotted_name":
        grandparent = getattr(parent, "parent", None)
        grandparent_type = getattr(grandparent, "type", None)
        if grandparent_type == "import_statement":
            return True
        if grandparent_type == "import_from_statement":
            name = _child_by_field(grandparent, "name")
            return name is not None and _node_contains(name, node)
    return False


def _is_definition_node(node) -> bool:
    """Best-effort tree-sitter Python definition check."""

    if node is None:
        return False
    node_type = getattr(node, "type", None)
    parent = getattr(node, "parent", None)
    parent_type = getattr(parent, "type", None)
    if node_type == "attribute":
        return _is_assignment_lhs(node, parent) if parent_type in {"assignment", "augmented_assignment"} else False
    if node_type not in {"identifier", "dotted_name"}:
        return False
    if parent_type in {"function_definition", "class_definition"}:
        return True
    if parent_type == "parameters":
        return _ancestor_has_type(parent, {"function_definition"})
    if parent_type in {"typed_parameter", "default_parameter", "pattern_list", "tuple_pattern", "list_pattern"}:
        return True
    if _is_import_alias(node):
        return True
    if parent_type in {"assignment", "augmented_assignment"}:
        return _is_assignment_lhs(node, parent)
    cur = parent
    depth = 0
    while cur is not None and depth < 6:
        cur_type = getattr(cur, "type", None)
        if cur_type in {"assignment", "augmented_assignment"}:
            return _is_assignment_lhs(node, cur)
        if cur_type == "for_statement":
            return _is_for_target(node, cur)
        if cur_type == "with_statement":
            return _is_with_alias(node, cur)
        cur = getattr(cur, "parent", None)
        depth += 1
    if parent_type == "for_statement":
        return _is_for_target(node, parent)
    if parent_type == "with_statement":
        return _is_with_alias(node, parent)
    return False


def _definition_node_for_token(node):
    if _is_definition_node(node):
        return node
    for child in _iter_nodes(node):
        if _is_definition_node(child):
            return child
    return None


def _parse_tree_sitter(source_bytes: bytes):
    if _PARSER is None:
        return None
    return _PARSER.parse(source_bytes)


def _type_from_python_ast(token: str) -> str:
    if token.isidentifier():
        return "identifier"
    if token.isnumeric():
        return "integer"
    return tokenize.tok_name.get(tokenize.OP, "token")


def _detect_var_defs_with_ast(source: str, token_spans: Sequence[TokenSpan], block_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback variable-definition detector based on Python's builtin AST."""

    num_blocks = (len(token_spans) + block_size - 1) // block_size
    block_mask = np.zeros(num_blocks, dtype=np.bool_)
    token_mask = np.zeros(len(token_spans), dtype=np.bool_)
    if num_blocks == 0:
        return block_mask, token_mask

    try:
        py_tree = ast.parse(source)
    except SyntaxError:
        return block_mask, token_mask

    line_starts = _line_start_byte_offsets(source)
    source_lines = source.splitlines(keepends=True) or [source]
    def_ranges: List[Tuple[int, int]] = []

    def add_range(start_line: int, start_col: int, end_line: int, end_col: int) -> None:
        start = _point_to_byte(line_starts, source_lines, (start_line - 1, start_col))
        end = _point_to_byte(line_starts, source_lines, (end_line - 1, end_col))
        def_ranges.append((start, end))

    def add_node_name(node) -> None:
        if hasattr(node, "lineno") and hasattr(node, "col_offset"):
            end_col = getattr(node, "end_col_offset", node.col_offset + 1)
            end_line = getattr(node, "end_lineno", node.lineno)
            add_range(node.lineno, node.col_offset, end_line, end_col)

    def add_function_or_class_name(node) -> None:
        line = source_lines[node.lineno - 1]
        keyword = "class" if isinstance(node, ast.ClassDef) else "def"
        keyword_col = line.find(keyword, node.col_offset)
        start_search = keyword_col + len(keyword) if keyword_col >= 0 else node.col_offset
        name_col = line.find(node.name, start_search)
        if name_col >= 0:
            add_range(node.lineno, name_col, node.lineno, name_col + len(node.name))

    def add_arg_names(args: ast.arguments) -> None:
        for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
            add_node_name(arg)
        if args.vararg is not None:
            add_node_name(args.vararg)
        if args.kwarg is not None:
            add_node_name(args.kwarg)

    def add_alias_name(alias: ast.alias, import_from: bool) -> None:
        local_name = alias.asname
        if local_name is None:
            local_name = alias.name if import_from else alias.name.split(".", 1)[0]
        if not local_name or not hasattr(alias, "lineno"):
            return
        line = source_lines[alias.lineno - 1]
        start_search = alias.col_offset
        if alias.asname is not None:
            as_pos = line.find(" as ", start_search)
            if as_pos >= 0:
                start_search = as_pos + 4
        name_col = line.find(local_name, start_search)
        if name_col >= 0:
            add_range(alias.lineno, name_col, alias.lineno, name_col + len(local_name))

    def add_attribute_name(node: ast.Attribute) -> None:
        if not hasattr(node, "lineno") or not hasattr(node, "end_lineno"):
            return
        if node.lineno != node.end_lineno:
            return
        line = source_lines[node.lineno - 1]
        end_col = getattr(node, "end_col_offset", node.col_offset + len(node.attr))
        search_end = min(len(line), end_col)
        attr_col = line.rfind("." + node.attr, node.col_offset, search_end)
        if attr_col >= 0:
            start_col = attr_col + 1
            add_range(node.lineno, start_col, node.lineno, start_col + len(node.attr))

    def visit_store_target(node) -> None:
        if isinstance(node, ast.Name):
            add_node_name(node)
        elif isinstance(node, ast.Attribute):
            add_attribute_name(node)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for item in node.elts:
                visit_store_target(item)
        elif isinstance(node, ast.Starred):
            visit_store_target(node.value)
        elif isinstance(node, ast.Subscript):
            return

    for node in ast.walk(py_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add_function_or_class_name(node)
            add_arg_names(node.args)
        elif isinstance(node, ast.Lambda):
            add_arg_names(node.args)
        elif isinstance(node, ast.ClassDef):
            add_function_or_class_name(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                add_alias_name(alias, import_from=False)
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                add_alias_name(alias, import_from=True)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                visit_store_target(target)
        elif isinstance(node, (ast.For, ast.AsyncFor)):
            visit_store_target(node.target)
        elif isinstance(node, ast.With):
            for item in node.items:
                if item.optional_vars is not None:
                    visit_store_target(item.optional_vars)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            line = source_lines[node.lineno - 1]
            name_col = line.find(node.name, node.col_offset)
            if name_col >= 0:
                add_range(node.lineno, name_col, node.lineno, name_col + len(node.name))
        elif isinstance(node, ast.NamedExpr):
            visit_store_target(node.target)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            add_node_name(node)
        elif isinstance(node, ast.arg):
            add_node_name(node)

    if not def_ranges:
        return block_mask, token_mask

    for tok_idx, (_, start, end) in enumerate(token_spans):
        if any(start < def_end and end > def_start for def_start, def_end in def_ranges):
            block_mask[tok_idx // block_size] = True
            token_mask[tok_idx] = True
    return block_mask, token_mask


class ASTFeatureExtractor:
    def __init__(
        self,
        config: ASTPreprocessConfig,
        vocab: Optional[TypeVocabulary] = None,
        tokenizer_name_or_path: Optional[str] = None,
    ) -> None:
        self.config = config
        self.vocab = vocab or TypeVocabulary()
        self.tokenizer_name_or_path = tokenizer_name_or_path
        self.tokenizer = load_hf_tokenizer(tokenizer_name_or_path) if tokenizer_name_or_path else None
        self.last_stats: Dict[str, object] = {}

    def tokenize(self, source: str) -> List[TokenSpan]:
        if self.tokenizer is not None:
            return tokenize_with_hf_offsets(source, self.tokenizer, self.config.max_tokens)
        return tokenize_python_source(source, self.config.max_tokens)

    def extract(self, source: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        token_spans = self.tokenize(source)
        num_tokens = len(token_spans)
        if num_tokens == 0:
            self.last_stats = {
                "num_tokens": 0,
                "num_blocks": 0,
                "tree_sitter_available": _PARSER is not None,
                "fallback_tokens": 0,
                "fallback_rate": 0.0,
                "fallback_token_mask": [],
            }
            return (
                np.zeros((0, self.config.block_size), dtype=np.int32),
                np.zeros((0,), dtype=np.bool_),
                np.zeros((0,), dtype=np.bool_),
            )

        num_blocks = (num_tokens + self.config.block_size - 1) // self.config.block_size
        ast_type_ids = np.full(
            (num_blocks, self.config.block_size),
            fill_value=self.config.pad_id,
            dtype=np.int32,
        )
        var_def_mask = np.zeros((num_blocks,), dtype=np.bool_)
        token_var_def_mask = np.zeros((num_tokens,), dtype=np.bool_)
        source_bytes = source.encode("utf-8")
        fallback_tokens = 0
        fallback_token_mask = np.zeros(num_tokens, dtype=np.bool_)

        if _PARSER is None:
            for idx, (tok, _, _) in enumerate(token_spans):
                ast_type_ids[idx // self.config.block_size, idx % self.config.block_size] = self.vocab.id_for(
                    _type_from_python_ast(tok)
                )
                fallback_tokens += 1
                fallback_token_mask[idx] = True
            var_def_mask, token_var_def_mask = _detect_var_defs_with_ast(source, token_spans, self.config.block_size)
            self.last_stats = {
                "num_tokens": num_tokens,
                "num_blocks": num_blocks,
                "tree_sitter_available": False,
                "fallback_tokens": fallback_tokens,
                "fallback_rate": fallback_tokens / max(1, num_tokens),
                "fallback_token_mask": fallback_token_mask.tolist(),
            }
            return ast_type_ids, var_def_mask, token_var_def_mask

        full_tree = None if self.config.prefix_parse else _parse_tree_sitter(source_bytes)

        for block_idx in range(num_blocks):
            start_tok = block_idx * self.config.block_size
            end_tok = min(start_tok + self.config.block_size, num_tokens)
            if self.config.prefix_parse:
                prefix_end = token_spans[end_tok - 1][2]
                tree = _parse_tree_sitter(source_bytes[:prefix_end])
            else:
                tree = full_tree

            if tree is None:
                for tok_idx in range(start_tok, end_tok):
                    tok, _, _ = token_spans[tok_idx]
                    ast_type_ids[block_idx, tok_idx - start_tok] = self.vocab.id_for(_type_from_python_ast(tok))
                    fallback_tokens += 1
                    fallback_token_mask[tok_idx] = True
                continue

            root = tree.root_node
            for tok_idx in range(start_tok, end_tok):
                tok, start, end = token_spans[tok_idx]
                node = root.descendant_for_byte_range(start, max(start + 1, end))
                node_type = getattr(node, "type", None) or "<UNKNOWN>"
                if node is None or _node_or_ancestor_is_error(node):
                    node_type = _type_from_python_ast(tok)
                    fallback_tokens += 1
                    fallback_token_mask[tok_idx] = True
                ast_type_ids[block_idx, tok_idx - start_tok] = self.vocab.id_for(node_type)
                if node is not None and _definition_node_for_token(node) is not None:
                    var_def_mask[block_idx] = True
                    token_var_def_mask[tok_idx] = True

        if not self.config.prefix_parse:
            ast_block_mask, ast_token_mask = _detect_var_defs_with_ast(source, token_spans, self.config.block_size)
            if bool(ast_token_mask.any()):
                var_def_mask = ast_block_mask
                token_var_def_mask = ast_token_mask

        self.last_stats = {
            "num_tokens": num_tokens,
            "num_blocks": num_blocks,
            "tree_sitter_available": True,
            "fallback_tokens": fallback_tokens,
            "fallback_rate": fallback_tokens / max(1, num_tokens),
            "fallback_token_mask": fallback_token_mask.tolist(),
        }
        return ast_type_ids, var_def_mask, token_var_def_mask


def iter_source_files(source_file: Optional[Path], source_dir: Optional[Path], pattern: str) -> Iterable[Path]:
    if source_file is not None:
        yield source_file
        return
    if source_dir is None:
        raise ValueError("Either --source-file or --source-dir must be provided.")
    yield from sorted(path for path in source_dir.glob(pattern) if path.is_file())


def write_sample(output_dir: Path, sample_name: str, ast_ids: np.ndarray, var_mask: np.ndarray, token_var_mask: np.ndarray) -> None:
    np.save(output_dir / f"{sample_name}_ast_ids.npy", ast_ids.astype(np.int32, copy=False))
    np.save(output_dir / f"{sample_name}_ast_mask.npy", var_mask.astype(np.bool_, copy=False))
    np.save(output_dir / f"{sample_name}_token_ast_mask.npy", token_var_mask.astype(np.bool_, copy=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build compact AST ids and variable-definition masks.")
    parser.add_argument("--source-file", type=Path)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--glob", type=str, default="**/*.py")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument(
        "--tokenizer-name-or-path",
        type=str,
        default=None,
        help="HuggingFace fast tokenizer used for model tokens. Strongly recommended for training artifacts.",
    )
    parser.add_argument(
        "--per-token-prefix",
        action="store_true",
        help="Parse a prefix ending at each token. Slow, but avoids block-internal future leakage.",
    )
    parser.add_argument(
        "--block-prefix-parse",
        action="store_true",
        help="Parse prefixes ending at block boundaries. Useful for leakage studies; offline training defaults to full parse.",
    )
    parser.add_argument("--encoding", type=str, default="utf-8")
    args = parser.parse_args()
    per_token_prefix = bool(args.per_token_prefix)
    if per_token_prefix and args.block_prefix_parse:
        parser.error("--per-token-prefix cannot be combined with --block-prefix-parse.")
    if per_token_prefix and _PARSER is None:
        parser.error("--per-token-prefix requires tree-sitter and tree-sitter-python.")

    config = ASTPreprocessConfig(
        block_size=args.block_size,
        max_tokens=args.max_tokens,
        prefix_parse=bool(args.block_prefix_parse or per_token_prefix),
    )
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    vocab = TypeVocabulary()
    extractor = ASTFeatureExtractor(config, vocab, tokenizer_name_or_path=args.tokenizer_name_or_path)
    processed = 0

    for path in iter_source_files(args.source_file, args.source_dir, args.glob):
        try:
            source = path.read_text(encoding=args.encoding)
        except UnicodeDecodeError:
            source = path.read_text(encoding=args.encoding, errors="replace")
        ast_ids, var_mask, token_var_mask = extractor.extract(source)
        if per_token_prefix:
            # Slow correction pass: each token sees only its own prefix.
            token_spans = extractor.tokenize(source)
            source_bytes = source.encode("utf-8")
            per_token_fallback = 0
            var_mask[:] = False
            token_var_mask[:] = False
            for tok_idx, (tok, start, end) in enumerate(token_spans):
                tree = _parse_tree_sitter(source_bytes[:end])
                if tree is None:
                    ast_ids[tok_idx // args.block_size, tok_idx % args.block_size] = vocab.id_for(_type_from_python_ast(tok))
                    per_token_fallback += 1
                    continue
                node = tree.root_node.descendant_for_byte_range(start, max(start + 1, end))
                if node is None or _node_or_ancestor_is_error(node):
                    ast_ids[tok_idx // args.block_size, tok_idx % args.block_size] = vocab.id_for(_type_from_python_ast(tok))
                    per_token_fallback += 1
                    continue
                ast_ids[tok_idx // args.block_size, tok_idx % args.block_size] = vocab.id_for(
                    getattr(node, "type", None) or "<UNKNOWN>"
                )
                if node is not None and _definition_node_for_token(node) is not None:
                    token_var_mask[tok_idx] = True
                    var_mask[tok_idx // args.block_size] = True
            extractor.last_stats["fallback_tokens"] = per_token_fallback
            extractor.last_stats["fallback_rate"] = per_token_fallback / max(1, len(token_spans))
        sample_name = path.stem if args.source_file else str(processed)
        write_sample(output_dir, sample_name, ast_ids, var_mask, token_var_mask)
        processed += 1

    metadata = {
        "config": asdict(config),
        "num_samples": processed,
        "tree_sitter_available": _PARSER is not None,
        "tokenizer_name_or_path": args.tokenizer_name_or_path,
        "token_alignment": "hf_offset_mapping" if args.tokenizer_name_or_path else "python_lexical_tokens",
        "per_token_prefix": per_token_prefix,
        "block_prefix_parse": bool(args.block_prefix_parse),
        "prefix_parse_note": (
            "per-token prefix parsing" if per_token_prefix else
            "block-prefix parsing; tokens inside a block may see later tokens from the same block"
            if args.block_prefix_parse else
            "complete-file offline parsing; use prefix diagnostics to measure inference degradation"
        ),
        "type_vocab": vocab.to_json(),
    }
    (output_dir / "ast_vocab.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"processed={processed} output_dir={os.fspath(output_dir)} vocab_size={len(vocab.type_to_id)}")


if __name__ == "__main__":
    main()
