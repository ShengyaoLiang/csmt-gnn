"""Direct structural-hallucination checks for generated code snippets.

The metric here is intentionally static and small.  It checks whether generated
or candidate code preserves the dependencies that the diagnostic case requires:
shadowed local variables, valid imports, and guarded attribute access.  This is
not a semantic proof of program correctness; it is a direct falsification probe
for the structural failures discussed in the paper.
"""

from __future__ import annotations

import argparse
import ast
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


@dataclass(frozen=True)
class StructuralCase:
    name: str
    source: str
    required_imports: Sequence[str]
    required_local_bindings: Sequence[str]
    required_guarded_attributes: Sequence[str]


@dataclass(frozen=True)
class CaseResult:
    name: str
    parse_ok: bool
    dependency_total: int
    dependency_preserved: int
    dependency_preservation_rate: float
    import_total: int
    import_preserved: int
    scope_total: int
    scope_preserved: int
    guard_total: int
    guard_preserved: int
    issues: List[str]


CASES: List[StructuralCase] = [
    StructuralCase(
        name="shadowing",
        source="""
value = 3
def scale(items):
    value = 10
    return [value * item for item in items]
result = scale([1, 2, 3])
""".strip(),
        required_imports=(),
        required_local_bindings=("scale:value", "scale:items"),
        required_guarded_attributes=(),
    ),
    StructuralCase(
        name="long_range_import",
        source="""
import math

def area(radius):
    padding = 0
    for _ in range(8):
        padding += 1
    return math.pi * radius * radius
""".strip(),
        required_imports=("math",),
        required_local_bindings=("area:radius",),
        required_guarded_attributes=(),
    ),
    StructuralCase(
        name="guarded_attribute",
        source="""
class Box:
    def __init__(self, value):
        self.value = value

def read(box):
    if box is not None:
        return box.value
    return 0
""".strip(),
        required_imports=(),
        required_local_bindings=("read:box",),
        required_guarded_attributes=("box.value",),
    ),
]


def parse_source(source: str) -> Optional[ast.AST]:
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.asname or alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                names.add(alias.asname or alias.name)
    return names


def function_defs(tree: ast.AST) -> Dict[str, ast.FunctionDef | ast.AsyncFunctionDef]:
    return {
        node.name: node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }


def local_bindings(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    names = {arg.arg for arg in fn.args.posonlyargs + fn.args.args + fn.args.kwonlyargs}
    if fn.args.vararg is not None:
        names.add(fn.args.vararg.arg)
    if fn.args.kwarg is not None:
        names.add(fn.args.kwarg.arg)
    for node in ast.walk(fn):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            names.add(node.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            names.add(node.name)
    return names


def attribute_name(node: ast.AST) -> Optional[str]:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = attribute_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    return None


def expression_mentions_name(node: ast.AST, name: str) -> bool:
    return any(isinstance(child, ast.Name) and child.id == name for child in ast.walk(node))


def guard_covers_attribute(fn: ast.FunctionDef | ast.AsyncFunctionDef, attr: str) -> bool:
    base = attr.split(".", 1)[0]
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        if not expression_mentions_name(node.test, base):
            continue
        for child in ast.walk(ast.Module(body=node.body, type_ignores=[])):
            if isinstance(child, ast.Attribute) and attribute_name(child) == attr:
                return True
    return False


def evaluate_case(case: StructuralCase, source: Optional[str] = None) -> CaseResult:
    text = case.source if source is None else source
    tree = parse_source(text)
    issues: List[str] = []
    if tree is None:
        total = len(case.required_imports) + len(case.required_local_bindings) + len(case.required_guarded_attributes)
        return CaseResult(
            name=case.name,
            parse_ok=False,
            dependency_total=total,
            dependency_preserved=0,
            dependency_preservation_rate=0.0,
            import_total=len(case.required_imports),
            import_preserved=0,
            scope_total=len(case.required_local_bindings),
            scope_preserved=0,
            guard_total=len(case.required_guarded_attributes),
            guard_preserved=0,
            issues=["syntax_error"],
        )

    imports = imported_names(tree)
    funcs = function_defs(tree)

    import_preserved = 0
    for name in case.required_imports:
        if name in imports:
            import_preserved += 1
        else:
            issues.append(f"missing_import:{name}")

    scope_preserved = 0
    for spec in case.required_local_bindings:
        fn_name, local = spec.split(":", 1)
        fn = funcs.get(fn_name)
        if fn is not None and local in local_bindings(fn):
            scope_preserved += 1
        else:
            issues.append(f"missing_local_binding:{spec}")

    guard_preserved = 0
    for attr in case.required_guarded_attributes:
        if any(guard_covers_attribute(fn, attr) for fn in funcs.values()):
            guard_preserved += 1
        else:
            issues.append(f"missing_guarded_attribute:{attr}")

    total = len(case.required_imports) + len(case.required_local_bindings) + len(case.required_guarded_attributes)
    preserved = import_preserved + scope_preserved + guard_preserved
    return CaseResult(
        name=case.name,
        parse_ok=True,
        dependency_total=total,
        dependency_preserved=preserved,
        dependency_preservation_rate=preserved / total if total else 1.0,
        import_total=len(case.required_imports),
        import_preserved=import_preserved,
        scope_total=len(case.required_local_bindings),
        scope_preserved=scope_preserved,
        guard_total=len(case.required_guarded_attributes),
        guard_preserved=guard_preserved,
        issues=issues,
    )


def load_candidate_map(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--candidates must be a JSON object keyed by case name")
    return {str(key): str(value) for key, value in data.items()}


def summarize(results: Iterable[CaseResult]) -> Dict[str, object]:
    rows = [asdict(result) for result in results]
    total = sum(int(row["dependency_total"]) for row in rows)
    preserved = sum(int(row["dependency_preserved"]) for row in rows)
    return {
        "purpose": "direct structural-hallucination dependency preservation check; not a benchmark",
        "dependency_total": total,
        "dependency_preserved": preserved,
        "dependency_preservation_rate": preserved / total if total else 1.0,
        "parse_success_rate": sum(1 for row in rows if row["parse_ok"]) / max(1, len(rows)),
        "cases": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate structural dependency preservation in generated code snippets.")
    parser.add_argument("--candidates", type=Path, help="Optional JSON object mapping diagnostic case names to generated code.")
    parser.add_argument("--output", type=Path, default=Path("results/structural_hallucination_eval.json"))
    args = parser.parse_args()

    candidate_map = load_candidate_map(args.candidates)
    results = [evaluate_case(case, candidate_map.get(case.name)) for case in CASES]
    summary = summarize(results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
