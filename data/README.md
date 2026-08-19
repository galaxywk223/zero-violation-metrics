# Aggregate Input

`metric_table.json` is the sanitized public input for the reported 54-cell matrix:

```text
6 methods x 3 environments x 3 seeds
```

Each row contains the method, environment, seed, completion status, training-step metadata, and episode-level aggregate metrics used by the generator. Checkpoint paths, policy files, simulator traces, logs, and machine-specific metadata are intentionally omitted.

The file is sufficient for rebuilding the repository's derived tables, figures, evidence summary, and paper outline. It does not support independent retraining or checkpoint-selection verification.
