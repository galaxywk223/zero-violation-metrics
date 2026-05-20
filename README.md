# Episode-Level Zero-Violation Metrics Supporting Evidence

This repository contains aggregate supporting evidence for an empirical study of episode-level zero-violation metrics in Safe Reinforcement Learning.

## Scope

The package summarizes a reported mature-baseline evaluation matrix:

```text
6 methods x 3 environments x 3 seeds x 5,000,000 training steps
```

The evaluated methods are `PPO`, `PPOLag`, `FOCOPS`, `CPO`, `CPPOPID`, and `PPOSaute`. The environments are `SafetyPointGoal1-v0`, `SafetyPointButton1-v0`, and `SafetyCarGoal1-v0`.

The repository scope is paper-facing and aggregate-only. It stores derived tables, figures, notes, and generation scripts. Primary simulator traces, policy parameters, large result directories, and machine-specific archives are excluded.

## Contents

- `tables/`: Markdown tables for method-level metrics, environment-method metrics, rankings, seed variability, metric correlations, bootstrap confidence intervals, relative-to-`PPO` trade-offs, main findings, literature positioning, paper positioning, literature-to-metric coverage, metric-family mapping, method trade-off quadrants, method safety signatures, reporting-protocol upgrade, environment case studies, environment-method scorecards, protocol coverage, claim flow, and claim boundaries.
- `figures/`: Paper figures for the metric protocol, literature positioning, paper positioning, literature-to-metric coverage, main findings, metric-family mapping, method trade-off quadrants, reporting-protocol upgrade, environment case studies, environment-method scorecards, protocol coverage, three-axis trade-off, environment zero-violation gap, method safety signatures, claim-evidence flow, environment metric profiles, reward-safety trade-offs, metric correlations, environment facets, seed variability, bootstrap intervals, Pareto frontier, method profiles, and claim boundaries.
- `notes/`: Evidence summary and supported / unsupported claim boundaries.
- `outline/`: Paper skeleton for the empirical study.
- `scripts/`: Artifact-generation script and lightweight validation test.
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

The generator expects a compatible aggregate metric-table archive containing `summaries/metric_table.json`. The aggregate metric table is sufficient to reproduce the derived paper tables and figures in this repository.

The aggregate archive is an external input and is intentionally excluded from the repository. Reproduction commands should pass the archive path explicitly with `--archive-path`. The repository does not require distribution of primary simulator traces, policy parameters, large result directories, or machine-specific archives. This boundary keeps the evidence small, inspectable, and aligned with the paper's empirical claim: a reporting and metric-separation claim over a reported baseline matrix.

## Anonymity and Release Boundary

Submissions should cite this package only as supporting material when venue policy permits non-identifying artifacts. Author-identifying repository URLs are intentionally omitted from manuscript text during submission review.

The public-release version is intended to contain:

- aggregate metric tables;
- generated paper figures;
- evidence notes and claim-boundary notes;
- deterministic generation and validation scripts;
- reproduction instructions for aggregate evidence.

The public-release version is not intended to contain:

- primary training-result directories;
- primary simulator traces;
- policy parameter files;
- machine-specific archives;
- system-specific paths;
- exploratory prototype results outside the reported matrix.

This boundary keeps the repository aligned with the manuscript contribution. The package supports an inspectable aggregate metric-reporting claim; it does not present additional hidden experiments or a new algorithm result.
