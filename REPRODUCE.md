# Aggregate Evidence Reproduction

## Environment

The aggregate-evidence generator uses standard Python data-analysis packages:

- `pandas`
- `numpy`
- `matplotlib`

No training framework is required to regenerate the derived tables and figures from the aggregate metric table.

## Input Contract

The generator expects an aggregate archive containing:

```text
summaries/metric_table.json
```

The archive is an external input and is not stored in this repository. Reproduction commands pass its path explicitly with `--archive-path`.

The table must contain 54 completed rows:

```text
6 methods x 3 environments x 3 seeds
```

Required fields include method, environment, seed, status, evaluation status, training budget, return, mean cost, safe rate, nonzero-cost frequency, tail costs, conditional unsafe severity, and maximum consecutive cost run.

## Validation

Run the structural check before rebuilding artifacts:

```powershell
python scripts/build_evidence_artifacts.py --archive-path <aggregate-metric-table-archive> --check-only
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
python scripts/build_evidence_artifacts.py --archive-path <aggregate-metric-table-archive> --repo-root .
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

## Submission Boundary

During submission review, repository URLs should be supplied only through venue-approved supporting-material mechanisms. The manuscript text should refer to aggregate reproducibility material without exposing author-identifying repository metadata.

The aggregate package is designed to support inspection of the reported evidence, not to serve as a hidden extension of the experiment section. Regeneration scripts operate on the aggregate metric-table contract and do not depend on primary execution directories.
