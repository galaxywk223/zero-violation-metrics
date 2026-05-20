# Paper Evidence Pack

## Purpose

The aggregate evaluation matrix provides the main evidence package for the Safe RL zero-violation metrics study. The evidence evaluates mature Safe RL baselines rather than a new-method positive result.

## Data Source

| field | value |
| --- | ---: |
| rows | 54 |
| methods | 6 |
| environments | 3 |
| seeds | 3 |
| successful runs | 54 |
| successful evaluations | 54 |
| training steps per run | 5000000 |

The result matrix is:

```text
6 methods x 3 environments x 3 seeds x 5,000,000 steps
```

## Supported Claims

1. Expected-cost safety and episode-level zero-violation safety are not identical empirical targets.
2. The evaluated mature baselines do not reach actual zero-violation behavior under the reported matrix.
3. PPO achieves the highest average return but has the weakest safety profile.
4. CPPOPID and PPOSaute achieve the strongest safe-rate profile but show large return loss.
5. FOCOPS remains the strongest balanced baseline for comparison with any future zero-violation-oriented method.
6. Zero-violation metrics expose safety differences that are not visible from mean cost alone.

## Unsupported Claims

1. The result does not prove that all Safe RL methods fail to achieve zero violation.
2. The result does not prove that any prototype zero-violation method is effective.
3. The result is not a new-method positive result.
4. The result does not justify restarting broad prototype-method training.
5. The result does not replace related-work positioning against chance-constrained and zero-violation Safe RL methods.

## Main Quantitative Findings

PPO has the highest overall return (`5.376`) but the weakest safety profile: safe rate `0.331` and nonzero-cost frequency `0.669`.

CPPOPID and PPOSaute are the strongest zero-violation-rate baselines. Their safe rates are `0.820` and `0.827`. Even the stronger of these still leaves an estimated `0.173` nonzero-cost episode frequency.

FOCOPS remains the main balanced baseline. It has return `3.063`, mean cost `5.842`, safe rate `0.653`, and conditional unsafe severity `15.976`.

Relative-to-PPO diagnostics make the reward-safety exchange explicit. FOCOPS retains `56.975`% of PPO return while reducing unsafe-episode frequency by `48.173`%. CPPOPID retains `20.496`% of PPO return and reduces unsafe episodes by `73.090`%. PPOSaute retains `21.628`% of PPO return and reduces unsafe episodes by `74.086`%.

The run-level return / safe-rate Pearson correlation is `-0.495`. The mean-cost / safe-rate Pearson correlation is `-0.886`. These diagnostic correlations support the paper framing that safety metrics should not be collapsed into a single expected-cost number.

Bootstrap intervals provide uncertainty context without adding new training. PPO return mean is `5.376` with 95% CI `[5.041, 5.742]`. FOCOPS safe-rate mean is `0.653` with 95% CI `[0.549, 0.749]`.

## Literature-to-Metric Coverage

The coverage map is a qualitative writing aid for related-work positioning. It shows that adjacent papers cover optimizers, benchmarks, implementation infrastructure, risk objectives, zero-violation methods, and statistical reliability, while the empirical study places all reporting dimensions in one baseline matrix.

| research line | expected cost | episode event | tail severity | persistence | protocol reliability |
| --- | ---: | ---: | ---: | ---: | ---: |
| Expected-cost Safe RL optimizers | 1.00 | 0.25 | 0.25 | 0.00 | 0.50 |
| Safety benchmark substrates | 0.75 | 0.50 | 0.25 | 0.00 | 1.00 |
| Implementation infrastructure | 0.50 | 0.25 | 0.00 | 0.00 | 1.00 |
| Risk and chance constraints | 0.50 | 0.75 | 1.00 | 0.25 | 0.50 |
| Zero or bounded violation methods | 0.50 | 1.00 | 0.50 | 0.25 | 0.50 |
| RL evaluation reliability | 0.25 | 0.25 | 0.25 | 0.25 | 1.00 |
| This empirical study | 1.00 | 1.00 | 1.00 | 1.00 | 0.75 |


## Ranking Summary

Mean rank across return and safety metrics:

| method | mean rank |
| --- | ---: |
| CPPOPID | 2.889 |
| PPOSaute | 3.056 |
| FOCOPS | 3.111 |
| CPO | 3.389 |
| PPOLag | 3.833 |
| PPO | 4.722 |

Safety-focused mean rank:

| method | mean rank |
| --- | ---: |
| CPPOPID | 2.333 |
| PPOSaute | 2.600 |
| FOCOPS | 3.133 |
| CPO | 3.667 |
| PPOLag | 3.800 |
| PPO | 5.467 |

## Main Interpretation

The aggregate result should anchor the empirical-study paper, not a new-method claim. Prototype zero-violation methods can remain future-work candidates only if the text clearly separates them from the baseline evidence.

## Generated Paper Artifacts

- `tables/method_overall_metrics.md`
- `tables/env_method_metrics.md`
- `tables/method_rankings.md`
- `tables/seed_variability.md`
- `tables/metric_correlations.md`
- `tables/bootstrap_ci.md`
- `tables/environment_wise_best_metrics.md`
- `tables/method_metric_rank_profile.md`
- `tables/relative_to_ppo_tradeoffs.md`
- `tables/claim_evidence_map.md`
- `tables/claim_boundary.md`
- `tables/main_findings_summary.md`
- `tables/literature_positioning_map.md`
- `tables/paper_positioning_matrix.md`
- `tables/literature_metric_coverage.md`
- `tables/metric_family_map.md`
- `tables/method_tradeoff_quadrants.md`
- `tables/method_safety_signature.md`
- `tables/claim_flow.md`
- `tables/key_numbers.md`
- `tables/metric_disagreement_summary.md`
- `tables/statistical_reporting_checklist.md`
- `tables/reporting_protocol_upgrade.md`
- `tables/environment_case_studies.md`
- `tables/env_method_scorecard.md`
- `tables/protocol_coverage_matrix.md`
- `figures/claim_aligned_main_evidence.png`
- `figures/metric_disagreement_summary.png`
- `figures/expected_cost_zero_violation_separation.png`
- `figures/statistical_reporting_ladder.png`
- `figures/reporting_protocol_upgrade.png`
- `figures/environment_case_studies.png`
- `figures/env_method_scorecard.png`
- `figures/protocol_coverage_matrix.png`
- `figures/return_vs_safe_rate.png`
- `figures/metric_protocol_schematic.png`
- `figures/literature_positioning_map.png`
- `figures/paper_positioning_matrix.png`
- `figures/literature_metric_coverage.png`
- `figures/core_takeaway_panel.png`
- `figures/mean_cost_vs_nonzero_frequency.png`
- `figures/tail_and_run_metrics.png`
- `figures/env_method_heatmap.png`
- `figures/pareto_frontier.png`
- `figures/seed_variability.png`
- `figures/metric_correlation_heatmap.png`
- `figures/bootstrap_confidence_intervals.png`
- `figures/env_tradeoff_facets.png`
- `figures/normalized_method_profiles.png`
- `figures/method_metric_rank_heatmap.png`
- `figures/zero_violation_gap_by_method.png`
- `figures/relative_to_ppo_tradeoffs.png`
- `figures/tradeoff_main_panel.png`
- `figures/claim_boundary.png`
- `figures/main_findings_summary.png`
- `figures/metric_family_map.png`
- `figures/method_tradeoff_quadrants.png`
- `figures/environment_metric_profiles.png`
- `figures/three_axis_tradeoff_bubble.png`
- `figures/env_zero_violation_gap_heatmap.png`
- `figures/method_safety_signature.png`
- `figures/claim_evidence_flow.png`

## Manuscript Integration Status

1. Formal figures are integrated into the English manuscript, Chinese manuscript, and PRICAI/LNCS submission entrypoints.
2. Related-work positioning covers Safe RL optimizers, benchmark and tooling substrates, risk and chance constraints, zero- or bounded-violation methods, and RL evaluation reliability.
3. The metric-separation argument distinguishes expected cost, zero-violation probability, tail severity, and temporal persistence as separate safety dimensions.
4. Literature-positioning artifacts frame the contribution as an evaluation and reporting layer rather than a new optimizer.
5. Remaining work is submission-specific polish: venue formatting, caption tightening, appendix selection, and supporting-material release hygiene.
