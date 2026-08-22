from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
from typing import Any
import json

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
BOOTSTRAP_SEED = 185
BOOTSTRAP_SAMPLES = 10_000
METHODS = ("PPO", "PPOLag", "FOCOPS", "CPO", "CPPOPID", "PPOSaute")
ENVS = ("SafetyPointGoal1-v0", "SafetyPointButton1-v0", "SafetyCarGoal1-v0")
BOOTSTRAP_METRICS = ("return", "safe_rate", "p95_cost")
CORRELATION_PAIRS = (
    ("mean_cost", "safe_rate"),
    ("safe_rate", "max_consecutive_cost_run"),
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build PRICAI camera-ready stratified statistics.")
    parser.add_argument("--input-path", type=Path, default=REPO_ROOT / "data" / "metric_table.json")
    parser.add_argument("--json-output", type=Path, default=REPO_ROOT / "tables" / "camera_ready_statistics.json")
    parser.add_argument("--markdown-output", type=Path, default=REPO_ROOT / "tables" / "camera_ready_statistics.md")
    parser.add_argument("--check-only", action="store_true")
    return parser


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Metric-table input must be a list")
    return payload


def prepare_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    required = {
        "method",
        "env_id",
        "seed",
        "return",
        "mean_cost",
        "safe_rate",
        "p95_cost",
        "conditional_unsafe_severity",
        "max_consecutive_cost_run",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing metric-table fields: {missing}")
    if len(frame) != 54:
        raise ValueError(f"Expected 54 rows, found {len(frame)}")
    observed = set(zip(frame["method"], frame["env_id"], frame["seed"], strict=True))
    expected = {(method, env_id, seed) for method in METHODS for env_id in ENVS for seed in (1, 2, 3)}
    if observed != expected:
        raise ValueError("Metric table does not contain the complete 6 x 3 x 3 matrix")
    return frame


def stratified_bootstrap(
    frame: pd.DataFrame,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for method in METHODS:
        method_frame = frame.loc[frame["method"] == method]
        for metric in BOOTSTRAP_METRICS:
            strata = [
                method_frame.loc[method_frame["env_id"] == env_id, metric].to_numpy(dtype=float)
                for env_id in ENVS
            ]
            draws = np.empty(samples, dtype=float)
            for idx in range(samples):
                env_means = [float(np.mean(rng.choice(values, size=len(values), replace=True))) for values in strata]
                draws[idx] = float(np.mean(env_means))
            rows.append(
                {
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "mean": float(method_frame[metric].mean()),
                    "method": method,
                    "metric": metric,
                    "resamples": samples,
                    "seed": seed,
                }
            )
    return rows


def correlation_rows(frame: pd.DataFrame, group_column: str, group_values: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for group in group_values:
        group_frame = frame.loc[frame[group_column] == group]
        for x_metric, y_metric in CORRELATION_PAIRS:
            rows.append(
                {
                    "group": group,
                    "group_type": "environment" if group_column == "env_id" else "method",
                    "n": int(len(group_frame)),
                    "pearson_r": float(group_frame[x_metric].corr(group_frame[y_metric])),
                    "x_metric": x_metric,
                    "y_metric": y_metric,
                }
            )
    return rows


def overall_rows(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    columns = (
        "return",
        "mean_cost",
        "safe_rate",
        "p95_cost",
        "conditional_unsafe_severity",
        "max_consecutive_cost_run",
    )
    grouped = frame.groupby("method", sort=False)[list(columns)].mean()
    return {
        method: {metric: float(grouped.loc[method, metric]) for metric in columns}
        for method in METHODS
    }


def ranking_disagreements(overall: dict[str, dict[str, float]]) -> list[dict[str, Any]]:
    return [
        {
            "interpretation": "PPOLag has the higher safe rate, whereas CPO has the lower mean cost and higher return.",
            "left_method": "PPOLag",
            "left_values": overall["PPOLag"],
            "right_method": "CPO",
            "right_values": overall["CPO"],
        },
        {
            "interpretation": "PPOSaute has the slightly higher safe rate, whereas CPPOPID has the lower mean and p95 cost.",
            "left_method": "CPPOPID",
            "left_values": overall["CPPOPID"],
            "right_method": "PPOSaute",
            "right_values": overall["PPOSaute"],
        },
    ]


def build_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    frame = prepare_dataframe(rows)
    overall = overall_rows(frame)
    return {
        "bootstrap": {
            "design": "Within each method, resample the three seeds separately inside each environment, then equally average the three environment means.",
            "rows": stratified_bootstrap(frame),
        },
        "correlations": {
            "environment": correlation_rows(frame, "env_id", ENVS),
            "method": correlation_rows(frame, "method", METHODS),
        },
        "overall": overall,
        "ranking_disagreements": ranking_disagreements(overall),
        "scope": {
            "environments": list(ENVS),
            "methods": list(METHODS),
            "rows": len(frame),
            "seeds": [1, 2, 3],
        },
    }


def fmt(value: float) -> str:
    return f"{value:.3f}"


def render_markdown(payload: dict[str, Any]) -> str:
    bootstrap = payload["bootstrap"]["rows"]
    correlations = payload["correlations"]
    disagreements = payload["ranking_disagreements"]
    lines = [
        "# Camera-Ready Statistics",
        "",
        "## Environment-Stratified Bootstrap",
        "",
        payload["bootstrap"]["design"],
        "",
        "| Method | Metric | Mean | 95% CI |",
        "| --- | --- | ---: | ---: |",
    ]
    for row in bootstrap:
        lines.append(
            f"| {row['method']} | {row['metric']} | {fmt(row['mean'])} | "
            f"[{fmt(row['ci95_low'])}, {fmt(row['ci95_high'])}] |"
        )
    lines.extend(["", "## Environment-Sliced Correlations", "", "| Environment | n | Pair | Pearson r |", "| --- | ---: | --- | ---: |"])
    for row in correlations["environment"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['x_metric']} vs. {row['y_metric']} | {fmt(row['pearson_r'])} |"
        )
    lines.extend(["", "## Method-Sliced Correlations", "", "| Method | n | Pair | Pearson r |", "| --- | ---: | --- | ---: |"])
    for row in correlations["method"]:
        lines.append(
            f"| {row['group']} | {row['n']} | {row['x_metric']} vs. {row['y_metric']} | {fmt(row['pearson_r'])} |"
        )
    lines.extend(["", "## Pairwise Ranking Disagreements", ""])
    for row in disagreements:
        lines.append(f"- **{row['left_method']} vs. {row['right_method']}:** {row['interpretation']}")
    lines.extend(
        [
            "",
            "All statistics are descriptive. The environment slices contain 18 method-seed cells and the method slices contain nine environment-seed cells.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = build_payload(read_rows(args.input_path.resolve()))
    print("camera_ready_statistics_valid=true")
    print(f"rows={payload['scope']['rows']}")
    print(f"bootstrap_rows={len(payload['bootstrap']['rows'])}")
    print(f"environment_correlation_rows={len(payload['correlations']['environment'])}")
    print(f"method_correlation_rows={len(payload['correlations']['method'])}")
    if args.check_only:
        print("check_only=true")
        return 0
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_markdown(payload), encoding="utf-8")
    print(f"json_output={args.json_output}")
    print(f"markdown_output={args.markdown_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
