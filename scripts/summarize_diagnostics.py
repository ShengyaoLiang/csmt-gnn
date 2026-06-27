"""Summarize tiny diagnostic runs across seeds.

The summary is a reporting helper for local mechanism checks. It does not turn
the diagnostic runs into a benchmark.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List


METRICS = (
    "final_loss",
    "eval_loss",
    "definition_use_loss",
    "cross_block_use_loss",
    "definition_use_preservation",
    "cross_block_use_preservation",
)


def load_run(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(variant: Dict, metric: str) -> float:
    if metric == "final_loss":
        return float(variant["final_loss"])
    value = variant["eval_metrics"].get(metric)
    if value is None:
        return float("nan")
    return float(value)


def format_mean_std(values: Iterable[float]) -> str:
    values = list(values)
    return f"{mean(values):.4f} ({pstdev(values):.4f})"


def summarize(paths: List[Path]) -> Dict:
    rows: Dict[str, Dict[str, List[float]]] = {}
    params: Dict[str, int] = {}
    seeds: List[int] = []
    random_variable_diffs: List[Dict[str, float]] = []

    for path in paths:
        run = load_run(path)
        seed = int(run["seed"])
        seeds.append(seed)
        by_variant = {item["variant"]: item for item in run["variants"]}
        for name, item in by_variant.items():
            params[name] = int(item["parameter_count"])
            rows.setdefault(name, {metric: [] for metric in METRICS})
            for metric in METRICS:
                value = metric_value(item, metric)
                if not math.isnan(value):
                    rows[name][metric].append(value)
        if "random_dropout_control" in by_variant and "variable_cvd" in by_variant:
            random_item = by_variant["random_dropout_control"]
            variable_item = by_variant["variable_cvd"]
            random_variable_diffs.append(
                {
                    "seed": seed,
                    "final_loss_delta_variable_minus_random": metric_value(variable_item, "final_loss")
                    - metric_value(random_item, "final_loss"),
                    "cross_block_delta_variable_minus_random": metric_value(variable_item, "cross_block_use_loss")
                    - metric_value(random_item, "cross_block_use_loss"),
                    "cross_block_preservation_delta_variable_minus_random": metric_value(
                        variable_item, "cross_block_use_preservation"
                    )
                    - metric_value(random_item, "cross_block_use_preservation"),
                }
            )

    variants = []
    for name in sorted(rows):
        metrics = rows[name]
        variants.append(
            {
                "variant": name,
                "params": params[name],
                "num_seeds": len(metrics["final_loss"]),
                "metrics": {
                    metric: {
                        "mean": mean(values),
                        "std": pstdev(values),
                        "values": values,
                    }
                    for metric, values in metrics.items()
                    if values and not any(math.isnan(value) for value in values)
                },
            }
        )

    cvd_summary = None
    if random_variable_diffs:
        final_diffs = [item["final_loss_delta_variable_minus_random"] for item in random_variable_diffs]
        cross_diffs = [item["cross_block_delta_variable_minus_random"] for item in random_variable_diffs]
        preservation_diffs = [
            item["cross_block_preservation_delta_variable_minus_random"]
            for item in random_variable_diffs
            if not math.isnan(item["cross_block_preservation_delta_variable_minus_random"])
        ]
        cvd_summary = {
            "seeds": random_variable_diffs,
            "final_loss_delta_variable_minus_random_mean": mean(final_diffs),
            "final_loss_delta_variable_minus_random_std": pstdev(final_diffs),
            "cross_block_delta_variable_minus_random_mean": mean(cross_diffs),
            "cross_block_delta_variable_minus_random_std": pstdev(cross_diffs),
        }
        if preservation_diffs:
            cvd_summary["cross_block_preservation_delta_variable_minus_random_mean"] = mean(preservation_diffs)
            cvd_summary["cross_block_preservation_delta_variable_minus_random_std"] = pstdev(preservation_diffs)

    return {
        "purpose": "seed-sweep summary for tiny diagnostics; not a benchmark",
        "seeds": sorted(seeds),
        "variants": variants,
        "random_vs_variable_cvd": cvd_summary,
    }


def markdown_table(summary: Dict) -> str:
    lines = [
        "| Variant | Seeds | Params | Final loss | Eval loss | Cross-block use loss | Cross-block preservation |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary["variants"]:
        metrics = row["metrics"]
        preservation = metrics.get("cross_block_use_preservation")
        preservation_text = format_mean_std(preservation["values"]) if preservation else "NA"
        lines.append(
            "| `{variant}` | {num_seeds} | {params} | {final_loss} | {eval_loss} | {cross} | {preservation} |".format(
                variant=row["variant"],
                num_seeds=row["num_seeds"],
                params=row["params"],
                final_loss=format_mean_std(metrics["final_loss"]["values"]),
                eval_loss=format_mean_std(metrics["eval_loss"]["values"]),
                cross=format_mean_std(metrics["cross_block_use_loss"]["values"]),
                preservation=preservation_text,
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tiny diagnostic JSON files.")
    parser.add_argument("--inputs", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("results/diagnostic_seed_sweep_1_5.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("results/diagnostic_seed_sweep_1_5.md"))
    args = parser.parse_args()

    summary = summarize(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    args.markdown_output.write_text(markdown_table(summary), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "markdown_output": str(args.markdown_output)}, indent=2))


if __name__ == "__main__":
    main()
