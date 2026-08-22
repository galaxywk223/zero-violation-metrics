from __future__ import annotations

from pathlib import Path
import json

from build_public_run_data import (
    EXPECTED_EPISODES,
    EXPECTED_EPISODES_PER_RUN,
    EXPECTED_RUNS,
    REQUIRED_RUN_FILES,
    REPO_ROOT,
    scan_private_text,
    validate_matrix,
)


METRIC_FIELDS = (
    "return",
    "mean_cost",
    "violation_rate",
    "safe_rate",
    "nonzero_cost_frequency",
    "p90_cost",
    "p95_cost",
    "max_cost",
    "conditional_unsafe_severity",
    "max_consecutive_cost_run",
)


def validate(repo_root: Path = REPO_ROOT) -> dict[str, int]:
    data_root = repo_root / "data"
    manifest = json.loads((data_root / "run_manifest.json").read_text(encoding="utf-8"))
    metric_rows = json.loads((data_root / "metric_table.json").read_text(encoding="utf-8"))
    metric_by_run = {row["run_id"]: row for row in metric_rows}
    run_roots = sorted(item for item in (data_root / "runs").iterdir() if item.is_dir())

    if len(run_roots) != EXPECTED_RUNS or len(metric_rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} runs and metric rows")
    if len(manifest.get("runs", [])) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} manifest entries")

    run_specs: list[dict[str, object]] = []
    episode_count = 0
    for run_root in run_roots:
        for name in REQUIRED_RUN_FILES:
            if not (run_root / name).is_file():
                raise ValueError(f"Missing {run_root.name}/{name}")

        evaluation = json.loads((run_root / "evaluation.json").read_text(encoding="utf-8"))
        summary = evaluation["summary"]
        for field in ("episode_returns", "episode_costs", "episode_lengths"):
            if len(summary[field]) != EXPECTED_EPISODES_PER_RUN:
                raise ValueError(f"{run_root.name}: expected 50 values for {field}")
        episode_count += len(summary["episode_costs"])

        metric_row = metric_by_run.get(run_root.name)
        if metric_row is None:
            raise ValueError(f"Missing metric row for {run_root.name}")
        for field in METRIC_FIELDS:
            if summary[field] != metric_row[field]:
                raise ValueError(f"{run_root.name}: {field} differs from metric table")
        run_specs.append(
            {
                "env_id": evaluation["env_id"],
                "method": evaluation["method"],
                "seed": evaluation["seed"],
            }
        )

    validate_matrix(run_specs)
    if episode_count != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} episodes, found {episode_count}")
    private_hits = scan_private_text(data_root)
    if private_hits:
        raise ValueError(f"Private text remains in public data: {private_hits[:10]}")

    return {
        "episodes": episode_count,
        "metric_rows": len(metric_rows),
        "private_text_hits": len(private_hits),
        "runs": len(run_roots),
    }


def main() -> int:
    summary = validate()
    for key, value in summary.items():
        print(f"public_{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
