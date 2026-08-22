# Code and Data Reproduction

## Environment

The public-data validator uses the Python standard library. The table and figure generator uses:

- `pandas`
- `numpy`
- `matplotlib`

Install the pinned dependency ranges in a clean environment:

```powershell
python -m pip install -r requirements.txt
```

## Public Data

The repository contains the complete reported run matrix:

```text
data/runs/<run_id>/config.json
data/runs/<run_id>/run_spec.json
data/runs/<run_id>/progress.csv
data/runs/<run_id>/evaluation.json
```

Each `evaluation.json` contains 50 episode returns, costs, lengths, and the corresponding summary metrics. `data/run_manifest.json` indexes all runs, while `data/metric_table.json` provides the aggregate generator input.

The table must contain 54 completed rows:

```text
6 methods x 3 environments x 3 seeds
```

## Run-Level Validation

Validate file coverage, the 2,700 episode records, metric-table consistency, and path sanitization:

```powershell
python scripts/validate_public_run_data.py
```

The expected summary is:

```text
public_episodes=2700
public_metric_rows=54
public_private_text_hits=0
public_runs=54
```

## Aggregate Validation and Regeneration

Validate the aggregate input:

```powershell
python scripts/build_evidence_artifacts.py --input-path data/metric_table.json --check-only
```

Regenerate all paper-facing aggregate evidence:

```powershell
python scripts/build_evidence_artifacts.py --input-path data/metric_table.json --repo-root .
```

Generate the camera-ready stratified statistics:

```powershell
python scripts/analyze_camera_ready_statistics.py
```

The command updates:

- `tables/`
- `figures/`
- `notes/`
- `outline/`

## Tests

Run the sanitizer and evidence-generator tests:

```powershell
python -m pytest scripts/test_build_public_run_data.py scripts/test_build_evidence_artifacts.py
```

## Training Code

The exact camera-ready training-code snapshot is [`04966dd601c3778e14638b5af82f498c4812f29d`](https://github.com/galaxywk223/zero-violation-paper-workbench/tree/04966dd601c3778e14638b5af82f498c4812f29d/experiments). The Round185 entrypoints are:

```text
experiments/scripts/run_cloud_batch_saferl_round185.py
experiments/scripts/watch_cloud_batch_round185.py
experiments/scripts/summarize_cloud_batch_round185.py
experiments/scripts/export_cloud_batch_round185.py
```

The complete sanitized execution export is attached to Release `pricai2026-camera-ready-data-v1` as `round185_run_level_artifacts_sanitized.zip`.
