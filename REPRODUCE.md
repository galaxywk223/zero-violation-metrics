# Aggregate Evidence Reproduction

## Environment

The aggregate-evidence generator uses standard Python data-analysis packages:

- `pandas`
- `numpy`
- `matplotlib`

No training framework is required to regenerate the derived tables and figures from the aggregate metric table.

Install the pinned dependency ranges in a clean environment:

```powershell
python -m pip install -r requirements.txt
```

## Input Contract

The public repository includes the sanitized aggregate input:

```text
data/metric_table.json
```

The generator also accepts an external archive containing `summaries/metric_table.json` for compatible private or historical inputs.

The table must contain 54 completed rows:

```text
6 methods x 3 environments x 3 seeds
```

Required fields include method, environment, seed, status, evaluation status, training budget, return, mean cost, safe rate, nonzero-cost frequency, tail costs, conditional unsafe severity, and maximum consecutive cost run.

## Validation

Run the structural check before rebuilding artifacts:

```powershell
python scripts/build_evidence_artifacts.py --input-path data/metric_table.json --check-only
```

The expected validation summary is:

```text
metric_archive_valid=true
rows=54
methods=6
envs=3
seeds=3
completed=54
completed_evaluations=54
check_only=true
```

## Regeneration

Regenerate all paper-facing aggregate evidence:

```powershell
python scripts/build_evidence_artifacts.py --input-path data/metric_table.json --repo-root .
```

The command updates:

- `tables/`
- `figures/`
- `notes/`
- `outline/`

## Lightweight Test

Run the included fake-data test:

```powershell
python scripts/test_build_evidence_artifacts.py
```

The test verifies that the generator can create tables, figures, notes, and outline files without reading or writing primary experiment-result directories.

## Evidence Boundary

The reproduction path regenerates aggregate paper evidence only. It does not retrain policies, rerun simulators, modify Safe RL libraries, or require policy parameter files.

## Public Evidence Boundary

The aggregate package supports inspection and regeneration of the reported evidence. It does not support independent retraining, checkpoint-selection verification, or recovery of primary run traces. Regeneration scripts operate on the aggregate metric-table contract and do not depend on primary execution directories.
