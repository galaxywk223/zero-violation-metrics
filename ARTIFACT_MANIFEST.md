# Artifact Manifest

## Public Run-Level Data

| Path | Role |
| --- | --- |
| `data/run_manifest.json` | Index and coverage summary for all 54 method-environment-seed cells. |
| `data/runs/<run_id>/config.json` | Recorded OmniSafe configuration for one training run. |
| `data/runs/<run_id>/run_spec.json` | Portable execution specification and artifact mapping. |
| `data/runs/<run_id>/progress.csv` | Full 5,000,000-step training curve. |
| `data/runs/<run_id>/evaluation.json` | Fifty episode returns, costs, lengths, and summary metrics. |
| `data/metric_table.json` | Sanitized 54-row aggregate input used to regenerate the paper-facing evidence. |

## Core Tables

| Path | Role |
| --- | --- |
| `tables/method_overall_metrics.md` | Aggregate method-level return, cost, safe-rate, tail, and run-length metrics. |
| `tables/env_method_metrics.md` | Environment-method slices used to inspect task-dependent behavior. |
| `tables/environment_wise_best_metrics.md` | Best method per environment and metric, used to show that metric choice changes method ordering. |
| `tables/relative_to_ppo_tradeoffs.md` | Effect-size view relative to the reward-only `PPO` baseline. |
| `tables/metric_correlations.md` | Run-level correlation matrix for return, mean cost, safe rate, nonzero-cost frequency, tail cost, and max run. |
| `tables/seed_variability.md` | Within-environment seed variability summary. |
| `tables/bootstrap_ci.md` | Bootstrap 95% confidence intervals over reported method runs. |
| `tables/main_findings_summary.md` | Condensed four-finding summary. |
| `tables/literature_positioning_map.md` | Literature-positioning map that separates evaluation-layer contribution from optimizer or guarantee claims. |
| `tables/paper_positioning_matrix.md` | Positioning matrix across optimizer, intervention, risk, guarantee, and benchmark paper families. |
| `tables/literature_metric_coverage.md` | Qualitative coverage map connecting adjacent paper families to expected-cost, episode-event, tail, persistence, and protocol-reliability dimensions. |
| `tables/metric_family_map.md` | Mapping from paper claims to reward, expected-cost, zero-violation, tail, and temporal-persistence metric families. |
| `tables/method_tradeoff_quadrants.md` | Method placement by return retention and unsafe-episode reduction relative to `PPO`. |
| `tables/method_safety_signature.md` | Normalized safety-facing method profile across frequency, severity, and persistence dimensions. |
| `tables/reporting_protocol_upgrade.md` | Reporting-protocol ladder from conventional return-cost summaries to episode-event, tail, persistence, and claim-boundary reporting. |
| `tables/environment_case_studies.md` | Environment-specific case-study table that identifies reward leaders, safety leaders, tail leaders, and residual zero-violation gaps. |
| `tables/env_method_scorecard.md` | Cell-level environment-method scorecard with safe rate, residual zero-violation gap, return, and tail metrics. |
| `tables/protocol_coverage_matrix.md` | Coverage and claim-boundary table for methods, tasks, runs, metrics, and artifacts. |
| `tables/claim_flow.md` | Paper-level chain from episode measurement to metric families, evidence, supported claims, and claim boundaries. |
| `tables/claim_boundary.md` | Claim boundary and unsupported readings. |
| `tables/claim_evidence_map.md` | Claim-to-evidence mapping for the reported matrix. |
| `tables/key_numbers.md` | Key reported numbers and their evidence roles. |
| `tables/method_metric_rank_profile.md` | Method rank profile across the reported metrics. |
| `tables/method_rankings.md` | Descriptive method rankings by aggregate metric families. |
| `tables/metric_disagreement_summary.md` | Summary of metric-dependent ordering differences. |
| `tables/statistical_reporting_checklist.md` | Checklist for descriptive uncertainty and claim boundaries. |

## Core Figures

| Path | Role |
| --- | --- |
| `figures/metric_protocol_schematic.png` | Evaluation protocol schematic from episode cost sequences to metric panel. |
| `figures/literature_positioning_map.png` | Visual map of adjacent Safe RL and RL evaluation lines. |
| `figures/paper_positioning_matrix.png` | Visual positioning matrix for the empirical reporting contribution. |
| `figures/literature_metric_coverage.png` | Heatmap showing the literature-to-metric coverage gap addressed by the reporting study. |
| `figures/core_takeaway_panel.png` | Four main takeaways in a compact visual panel. |
| `figures/metric_family_map.png` | Metric-family map separating reward, expected cost, zero-violation frequency, tail severity, and temporal persistence. |
| `figures/method_tradeoff_quadrants.png` | Return-retention versus unsafe-episode-reduction quadrant plot. |
| `figures/environment_metric_profiles.png` | Environment-level metric-family profile over the reported matrix. |
| `figures/three_axis_tradeoff_bubble.png` | Three-axis view of return, safe rate, and nonzero-cost episode frequency. |
| `figures/env_zero_violation_gap_heatmap.png` | Environment-method heatmap of residual zero-violation gap. |
| `figures/method_safety_signature.png` | Normalized method safety-signature heatmap. |
| `figures/reporting_protocol_upgrade.png` | Visual reporting ladder from return-cost reporting to episode-level zero-violation reporting. |
| `figures/environment_case_studies.png` | Three-environment case-study panel for task-dependent interpretation. |
| `figures/env_method_scorecard.png` | Full environment-method scorecard showing safe rate, return, and residual zero-violation gap in each task-method cell. |
| `figures/protocol_coverage_matrix.png` | Visual matrix pairing each coverage axis with its supported claim and boundary. |
| `figures/claim_evidence_flow.png` | Visual flow from measurement to metric families, evidence, claims, and boundaries. |
| `figures/return_vs_safe_rate.png` | Reward-safety trade-off scatter plot. |
| `figures/relative_to_ppo_tradeoffs.png` | Safety gains and return retention relative to `PPO`. |
| `figures/tradeoff_main_panel.png` | Two-panel reward-safety trade-off figure combining return-safe-rate placement with relative-to-`PPO` effect sizes. |
| `figures/claim_aligned_main_evidence.png` | Main-evidence panel aligned with the supported claim boundary. |
| `figures/env_method_heatmap.png` | Environment-method safety-rate heatmap. |
| `figures/expected_cost_zero_violation_separation.png` | Expected-cost versus zero-violation separation figure. |
| `figures/main_findings_summary.png` | Compact visual summary of the main empirical findings. |
| `figures/mean_cost_vs_nonzero_frequency.png` | Mean-cost and nonzero-frequency comparison. |
| `figures/metric_disagreement_summary.png` | Metric-ordering disagreement summary. |
| `figures/statistical_reporting_ladder.png` | Statistical-reporting ladder from point estimates to scoped uncertainty. |
| `figures/tail_and_run_metrics.png` | Tail-severity and temporal-persistence comparison. |
| `figures/metric_correlation_heatmap.png` | Metric-correlation heatmap showing non-equivalence among safety metrics. |
| `figures/env_tradeoff_facets.png` | Environment-sliced reward-safe-rate facets. |
| `figures/bootstrap_confidence_intervals.png` | Bootstrap uncertainty context for core metrics. |
| `figures/seed_variability.png` | Seed variability across method/environment cells. |
| `figures/pareto_frontier.png` | Return-safe-rate Pareto view over the reported matrix. |
| `figures/normalized_method_profiles.png` | Normalized method profiles across core metrics. |
| `figures/method_metric_rank_heatmap.png` | Rank heatmap showing metric-dependent ordering. |
| `figures/zero_violation_gap_by_method.png` | Gap from true zero violation by method. |
| `figures/claim_boundary.png` | Visual claim boundary. |

## Notes and Scripts

| Path | Role |
| --- | --- |
| `notes/evidence_summary.md` | Evidence-pack note with supported and unsupported claims. |
| `outline/zero_violation_empirical_study_skeleton.md` | Paper skeleton used to align narrative sections. |
| `scripts/build_evidence_artifacts.py` | Deterministic artifact generator from the aggregate metric table archive. |
| `scripts/test_build_evidence_artifacts.py` | Lightweight generator test using a temporary fake metric-table archive. |
| `scripts/build_public_run_data.py` | Deterministic sanitizer and Release-archive builder for the canonical Round185 export. |
| `scripts/validate_public_run_data.py` | Public validation of run coverage, episode records, metrics, and path sanitization. |
| `scripts/test_build_public_run_data.py` | Unit tests for path sanitization and portable run specifications. |

## Release Asset

Release `pricai2026-camera-ready-data-v1` contains `round185_run_level_artifacts_sanitized.zip`, the complete sanitized 439-file Round185 execution export, and `SHA256SUMS.txt`.
