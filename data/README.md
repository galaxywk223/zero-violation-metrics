# Public Run-Level Data

The directory contains the reported 54-cell matrix:

```text
6 methods x 3 environments x 3 seeds
```

- `runs/<run_id>/config.json`: OmniSafe configuration recorded for the run.
- `runs/<run_id>/run_spec.json`: Portable method, environment, seed, budget, and artifact definition.
- `runs/<run_id>/progress.csv`: Training curve recorded across the 5,000,000-step run.
- `runs/<run_id>/evaluation.json`: Deterministic 50-episode returns, costs, lengths, and summary metrics.
- `run_manifest.json`: Matrix coverage, counts, portable paths, and source checksum.
- `metric_table.json`: Aggregate 54-row input used to generate the paper tables and figures.

The 54 evaluation files contain 2,700 episode records. `python scripts/validate_public_run_data.py` verifies run coverage, episode counts, metric consistency, and path sanitization.
