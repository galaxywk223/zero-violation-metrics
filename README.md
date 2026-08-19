# Episode-Level Zero-Violation Metrics Supporting Evidence

This repository contains aggregate supporting evidence for an empirical study of episode-level zero-violation metrics in Safe Reinforcement Learning.

## Scope

The package summarizes a reported mature-baseline evaluation matrix:

```text
6 methods x 3 environments x 3 seeds x 5,000,000 training steps
```

The evaluated methods are `PPO`, `PPOLag`, `FOCOPS`, `CPO`, `CPPOPID`, and `PPOSaute`. The environments are `SafetyPointGoal1-v0`, `SafetyPointButton1-v0`, and `SafetyCarGoal1-v0`.

The repository scope is paper-facing and aggregate-only. It stores the sanitized 54-cell metric input, derived tables, figures, notes, and generation scripts. Primary simulator traces, policy parameters, large result directories, checkpoints, and machine-specific archives are excluded.

## Contents

- `tables/`: Markdown tables for method-level metrics, environment-method metrics, rankings, seed variability, metric correlations, bootstrap confidence intervals, relative-to-`PPO` trade-offs, main findings, literature positioning, paper positioning, literature-to-metric coverage, metric-family mapping, method trade-off quadrants, method safety signatures, reporting-protocol upgrade, environment case studies, environment-method scorecards, protocol coverage, claim flow, and claim boundaries.
- `figures/`: Paper figures for the metric protocol, literature positioning, paper positioning, literature-to-metric coverage, main findings, metric-family mapping, method trade-off quadrants, reporting-protocol upgrade, environment case studies, environment-method scorecards, protocol coverage, three-axis trade-off, environment zero-violation gap, method safety signatures, claim-evidence flow, environment metric profiles, reward-safety trade-offs, metric correlations, environment facets, seed variability, bootstrap intervals, Pareto frontier, method profiles, and claim boundaries.
- `data/metric_table.json`: Sanitized aggregate input containing the 54 reported method-environment-seed rows. Checkpoint paths and machine-specific metadata are omitted.
- `notes/`: Evidence summary and supported / unsupported claim boundaries.
- `outline/`: Paper skeleton for the empirical study.
- `scripts/`: Artifact-generation script and lightweight validation test.
- `requirements.txt`: Minimal Python dependencies for generation and validation.
- `ARTIFACT_MANIFEST.md`: File-level manifest and role of each artifact group.
- `REPRODUCE.md`: Regeneration and validation instructions.

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

## Evidence and Reproducibility Boundary

The generator accepts the checked-in `data/metric_table.json` directly or a compatible archive containing `summaries/metric_table.json`. The aggregate metric table is sufficient to reproduce the derived paper tables and figures in this repository.

The public input contains aggregate metrics only. The repository does not distribute primary simulator traces, policy parameters, checkpoints, large result directories, or machine-specific archives. This boundary keeps the evidence small, inspectable, and aligned with the paper's empirical claim: a reporting and metric-separation claim over a reported baseline matrix.

## Public Release Boundary

This repository is the public supporting-material package for the accepted PRICAI 2026 paper. It contains the aggregate evidence needed to inspect and regenerate the paper-facing tables, figures, and evidence summary.

The public package is not intended to contain:

- primary training-result directories;
- primary simulator traces;
- policy parameter files;
- machine-specific archives;
- system-specific paths;
- exploratory prototype results outside the reported matrix.

This boundary keeps the repository aligned with the manuscript contribution. The package supports an inspectable aggregate metric-reporting claim; it does not present additional hidden experiments or a new algorithm result.
