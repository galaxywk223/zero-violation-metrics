from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_camera_ready_statistics import (  # noqa: E402
    ENVS,
    METHODS,
    build_payload,
    prepare_dataframe,
    stratified_bootstrap,
)


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method_idx, method in enumerate(METHODS):
        for env_idx, env_id in enumerate(ENVS):
            for seed in (1, 2, 3):
                cost = float(method_idx + env_idx + seed)
                rows.append(
                    {
                        "conditional_unsafe_severity": cost + 2.0,
                        "env_id": env_id,
                        "max_consecutive_cost_run": cost + 3.0,
                        "mean_cost": cost,
                        "method": method,
                        "p95_cost": cost + 4.0,
                        "return": 20.0 - cost,
                        "safe_rate": 1.0 - cost / 20.0,
                        "seed": seed,
                    }
                )
    return rows


def test_prepare_dataframe_accepts_complete_matrix() -> None:
    frame = prepare_dataframe(make_rows())

    assert len(frame) == 54
    assert set(frame["method"]) == set(METHODS)


def test_stratified_bootstrap_is_deterministic() -> None:
    frame = pd.DataFrame(make_rows())

    first = stratified_bootstrap(frame, samples=100, seed=185)
    second = stratified_bootstrap(frame, samples=100, seed=185)

    assert first == second
    assert len(first) == len(METHODS) * 3


def test_payload_contains_all_slices_and_disagreements() -> None:
    payload = build_payload(make_rows())

    assert len(payload["correlations"]["environment"]) == len(ENVS) * 2
    assert len(payload["correlations"]["method"]) == len(METHODS) * 2
    assert len(payload["ranking_disagreements"]) == 2
