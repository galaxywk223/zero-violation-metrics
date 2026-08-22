# Episode-Level Zero-Violation Metrics: Code and Data

This repository contains code, run-level records, and derived evidence for an empirical study of episode-level zero-violation metrics in Safe Reinforcement Learning.

## Scope

The package summarizes a reported mature-baseline evaluation matrix:

```text
6 methods x 3 environments x 3 seeds x 5,000,000 training steps
```

The evaluated methods are `PPO`, `PPOLag`, `FOCOPS`, `CPO`, `CPPOPID`, and `PPOSaute`. The environments are `SafetyPointGoal1-v0`, `SafetyPointButton1-v0`, and `SafetyCarGoal1-v0`.

The public data cover all 54 method-environment-seed cells. Each run includes its configuration, training curve, and deterministic 50-episode evaluation record. The fixed training-code snapshot is available at [`zero-violation-paper-workbench@04966dd`](https://github.com/galaxywk223/zero-violation-paper-workbench/tree/04966dd601c3778e14638b5af82f498c4812f29d/experiments).

## Contents

- `tables/`: Markdown tables for method-level metrics, environment-method metrics, rankings, seed variability, metric correlations, bootstrap confidence intervals, relative-to-`PPO` trade-offs, main findings, literature positioning, paper positioning, literature-to-metric coverage, metric-family mapping, method trade-off quadrants, method safety signatures, reporting-protocol upgrade, environment case studies, environment-method scorecards, protocol coverage, claim flow, and claim boundaries.
- `figures/`: Paper figures for the metric protocol, literature positioning, paper positioning, literature-to-metric coverage, main findings, metric-family mapping, method trade-off quadrants, reporting-protocol upgrade, environment case studies, environment-method scorecards, protocol coverage, three-axis trade-off, environment zero-violation gap, method safety signatures, claim-evidence flow, environment metric profiles, reward-safety trade-offs, metric correlations, environment facets, seed variability, bootstrap intervals, Pareto frontier, method profiles, and claim boundaries.
- `data/runs/`: Portable configuration, training-curve, and per-episode evaluation files for all 54 runs.
- `data/run_manifest.json`: Matrix definition, artifact paths, counts, and source archive checksum.
- `data/metric_table.json`: Sanitized 54-row aggregate input used by the evidence generator.
- `notes/`: Evidence summary and supported / unsupported claim boundaries.
- `outline/`: Paper skeleton for the empirical study.
- `scripts/`: Run-data sanitizer, public-data validator, artifact generator, and tests.
- `tables/camera_ready_statistics.{md,json}`: Stratified uncertainty, sliced correlations, and pairwise ranking diagnostics used in the camera-ready revision.
- `requirements.txt`: Minimal Python dependencies for generation and validation.
- `ARTIFACT_MANIFEST.md`: File-level manifest and role of each artifact group.
- `REPRODUCE.md`: Regeneration and validation instructions.
- [Release `pricai2026-camera-ready-data-v1`](https://github.com/galaxywk223/zero-violation-metrics/releases/tag/pricai2026-camera-ready-data-v1): Complete sanitized Round185 export and SHA-256 checksum.

## Claim Boundary

Supported claims:

- Expected-cost safety and episode-level zero-violation safety are distinct empirical targets.
- The evaluated mature baselines do not reach true zero-violation behavior in the reported matrix.
- `PPO` is high-return but unsafe under episode-level metrics.
- `CPPOPID` and `PPOSaute` reach higher safe rates but with large return loss.
- `FOCOPS` is the main balanced baseline comparator.

Unsupported claims:

- The matrix does not prove that all Safe RL methods fail.
- The matrix does not validate a new algorithm.
- The matrix does not prove that any prototype zero-violation method is effective.
- The matrix does not establish a universal Pareto frontier for Safe RL.

## Validation and Regeneration

Validate the public run-level records:

```powershell
python scripts/validate_public_run_data.py
```

Regenerate the reported tables and figures:

```powershell
python scripts/build_evidence_artifacts.py --input-path data/metric_table.json --repo-root .
```

Detailed environment, training, validation, and testing commands are defined in `REPRODUCE.md`.
