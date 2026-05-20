from pathlib import Path
import json
import sys
import tempfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_evidence_artifacts import (  # noqa: E402
    EXPECTED_ENVS,
    EXPECTED_METHODS,
    EXPECTED_SEEDS,
    build_artifacts,
    read_metric_rows,
    validate_rows,
)


def make_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for env_idx, env_id in enumerate(EXPECTED_ENVS):
        for method_idx, method in enumerate(EXPECTED_METHODS):
            for seed in EXPECTED_SEEDS:
                cost = 2.0 + env_idx + method_idx * 0.5 + seed * 0.1
                safe_rate = max(0.1, 0.9 - method_idx * 0.08 - env_idx * 0.03)
                rows.append(
                    {
                        "run_id": f"{method.lower()}__{env_id}__seed{seed:03d}",
                        "method": method,
                        "env_id": env_id,
                        "seed": seed,
                        "status": "completed",
                        "training_executed": True,
                        "evaluation_status": "completed",
                        "requested_steps": 5_000_000,
                        "actual_steps": 5_000_000,
                        "error": None,
                        "return": 6.0 - method_idx * 0.4 + env_idx * 0.2,
                        "mean_cost": cost,
                        "violation_rate": cost / 200.0,
                        "safe_rate": safe_rate,
                        "nonzero_cost_frequency": 1.0 - safe_rate,
                        "p90_cost": cost + 4.0,
                        "p95_cost": cost + 5.0,
                        "max_cost": cost + 8.0,
                        "conditional_unsafe_severity": cost + 2.0,
                        "max_consecutive_cost_run": 10 + method_idx + env_idx,
                    }
                )
    return rows


def make_metric_archive(path: Path, rows: list[dict[str, object]]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("summaries/metric_table.json", json.dumps(rows))


def test_validate_rows_accepts_complete_matrix():
    rows = make_rows()
    summary = validate_rows(rows)

    assert summary["rows"] == 54
    assert summary["completed"] == 54
    assert summary["completed_evaluations"] == 54


def test_read_metric_rows_uses_aggregate_archive_contract():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        archive_path = root / "metrics.archive"
        rows = make_rows()
        make_metric_archive(archive_path, rows)

        loaded = read_metric_rows(archive_path)

    assert len(loaded) == 54
    assert loaded[0]["status"] == "completed"


def test_build_artifacts_writes_paper_pack_without_results_dir():
    with tempfile.TemporaryDirectory() as tmp:
        repo_root = Path(tmp)
        rows = make_rows()
        validation = validate_rows(rows)

        build_artifacts(rows, repo_root, validation)

        assert (repo_root / "tables" / "method_overall_metrics.md").exists()
        assert (repo_root / "tables" / "env_method_metrics.md").exists()
        assert (repo_root / "tables" / "method_rankings.md").exists()
        assert (repo_root / "tables" / "seed_variability.md").exists()
        assert (repo_root / "tables" / "metric_correlations.md").exists()
        assert (repo_root / "tables" / "bootstrap_ci.md").exists()
        assert (repo_root / "tables" / "environment_wise_best_metrics.md").exists()
        assert (repo_root / "tables" / "method_metric_rank_profile.md").exists()
        assert (repo_root / "tables" / "relative_to_ppo_tradeoffs.md").exists()
        assert (repo_root / "tables" / "claim_evidence_map.md").exists()
        assert (repo_root / "tables" / "claim_boundary.md").exists()
        assert (repo_root / "tables" / "main_findings_summary.md").exists()
        assert (repo_root / "tables" / "literature_positioning_map.md").exists()
        assert (repo_root / "tables" / "paper_positioning_matrix.md").exists()
        assert (repo_root / "tables" / "literature_metric_coverage.md").exists()
        assert (repo_root / "tables" / "metric_family_map.md").exists()
        assert (repo_root / "tables" / "method_tradeoff_quadrants.md").exists()
        assert (repo_root / "tables" / "method_safety_signature.md").exists()
        assert (repo_root / "tables" / "claim_flow.md").exists()
        assert (repo_root / "tables" / "key_numbers.md").exists()
        assert (repo_root / "tables" / "metric_disagreement_summary.md").exists()
        assert (repo_root / "tables" / "statistical_reporting_checklist.md").exists()
        assert (repo_root / "tables" / "reporting_protocol_upgrade.md").exists()
        assert (repo_root / "tables" / "environment_case_studies.md").exists()
        assert (repo_root / "tables" / "env_method_scorecard.md").exists()
        assert (repo_root / "tables" / "protocol_coverage_matrix.md").exists()
        assert (repo_root / "figures" / "return_vs_safe_rate.png").exists()
        assert (repo_root / "figures" / "metric_protocol_schematic.png").exists()
        assert (repo_root / "figures" / "literature_positioning_map.png").exists()
        assert (repo_root / "figures" / "paper_positioning_matrix.png").exists()
        assert (repo_root / "figures" / "literature_metric_coverage.png").exists()
        assert (repo_root / "figures" / "core_takeaway_panel.png").exists()
        assert (repo_root / "figures" / "mean_cost_vs_nonzero_frequency.png").exists()
        assert (repo_root / "figures" / "tail_and_run_metrics.png").exists()
        assert (repo_root / "figures" / "env_method_heatmap.png").exists()
        assert (repo_root / "figures" / "pareto_frontier.png").exists()
        assert (repo_root / "figures" / "seed_variability.png").exists()
        assert (repo_root / "figures" / "metric_correlation_heatmap.png").exists()
        assert (repo_root / "figures" / "bootstrap_confidence_intervals.png").exists()
        assert (repo_root / "figures" / "env_tradeoff_facets.png").exists()
        assert (repo_root / "figures" / "normalized_method_profiles.png").exists()
        assert (repo_root / "figures" / "method_metric_rank_heatmap.png").exists()
        assert (repo_root / "figures" / "zero_violation_gap_by_method.png").exists()
        assert (repo_root / "figures" / "relative_to_ppo_tradeoffs.png").exists()
        assert (repo_root / "figures" / "tradeoff_main_panel.png").exists()
        assert (repo_root / "figures" / "claim_boundary.png").exists()
        assert (repo_root / "figures" / "main_findings_summary.png").exists()
        assert (repo_root / "figures" / "metric_family_map.png").exists()
        assert (repo_root / "figures" / "method_tradeoff_quadrants.png").exists()
        assert (repo_root / "figures" / "environment_metric_profiles.png").exists()
        assert (repo_root / "figures" / "three_axis_tradeoff_bubble.png").exists()
        assert (repo_root / "figures" / "env_zero_violation_gap_heatmap.png").exists()
        assert (repo_root / "figures" / "method_safety_signature.png").exists()
        assert (repo_root / "figures" / "claim_evidence_flow.png").exists()
        assert (repo_root / "figures" / "claim_aligned_main_evidence.png").exists()
        assert (repo_root / "figures" / "metric_disagreement_summary.png").exists()
        assert (repo_root / "figures" / "expected_cost_zero_violation_separation.png").exists()
        assert (repo_root / "figures" / "statistical_reporting_ladder.png").exists()
        assert (repo_root / "figures" / "reporting_protocol_upgrade.png").exists()
        assert (repo_root / "figures" / "environment_case_studies.png").exists()
        assert (repo_root / "figures" / "env_method_scorecard.png").exists()
        assert (repo_root / "figures" / "protocol_coverage_matrix.png").exists()
        evidence = repo_root / "notes" / "evidence_summary.md"
        assert evidence.exists()
        text = evidence.read_text(encoding="utf-8")
        assert "Unsupported Claims" in text
        assert "does not prove that any prototype zero-violation method is effective" in text
        assert "claim_evidence_map.md" in text
        assert "metric_correlation_heatmap.png" in text
        assert "metric_protocol_schematic.png" in text
        assert "core_takeaway_panel.png" in text
        assert "bootstrap_confidence_intervals.png" in text
        assert "env_tradeoff_facets.png" in text
        assert "method_metric_rank_heatmap.png" in text
        assert "zero_violation_gap_by_method.png" in text
        assert "relative_to_ppo_tradeoffs.png" in text
        assert "tradeoff_main_panel.png" in text
        assert "relative_to_ppo_tradeoffs.md" in text
        assert "claim_boundary.md" in text
        assert "claim_boundary.png" in text
        assert "main_findings_summary.md" in text
        assert "main_findings_summary.png" in text
        assert "literature_positioning_map.md" in text
        assert "literature_positioning_map.png" in text
        assert "paper_positioning_matrix.md" in text
        assert "paper_positioning_matrix.png" in text
        assert "literature_metric_coverage.md" in text
        assert "literature_metric_coverage.png" in text
        assert "metric_family_map.md" in text
        assert "metric_family_map.png" in text
        assert "method_tradeoff_quadrants.md" in text
        assert "method_tradeoff_quadrants.png" in text
        assert "environment_metric_profiles.png" in text
        assert "method_safety_signature.md" in text
        assert "claim_flow.md" in text
        assert "three_axis_tradeoff_bubble.png" in text
        assert "env_zero_violation_gap_heatmap.png" in text
        assert "method_safety_signature.png" in text
        assert "claim_evidence_flow.png" in text
        assert "key_numbers.md" in text
        assert "metric_disagreement_summary.md" in text
        assert "statistical_reporting_checklist.md" in text
        assert "claim_aligned_main_evidence.png" in text
        assert "metric_disagreement_summary.png" in text
        assert "expected_cost_zero_violation_separation.png" in text
        assert "statistical_reporting_ladder.png" in text
        assert "reporting_protocol_upgrade.md" in text
        assert "environment_case_studies.md" in text
        assert "env_method_scorecard.md" in text
        assert "protocol_coverage_matrix.md" in text
        assert "reporting_protocol_upgrade.png" in text
        assert "environment_case_studies.png" in text
        assert "env_method_scorecard.png" in text
        assert "protocol_coverage_matrix.png" in text
        assert "Relative-to-PPO diagnostics" in text
        assert "Bootstrap intervals provide uncertainty context" in text
        assert "Manuscript Integration Status" in text
        assert "Required Follow-Up" not in text
        assert "Convert the generated figures" not in text
        boundary = (repo_root / "tables" / "claim_boundary.md").read_text(encoding="utf-8")
        assert "All Safe RL methods fail under all settings" in boundary
        assert "successful prototype zero-violation algorithm" in boundary
        findings = (repo_root / "tables" / "main_findings_summary.md").read_text(encoding="utf-8")
        assert "Reward-first optimization leaves frequent unsafe episodes" in findings
        assert "Safety metrics are coupled but not interchangeable" in findings
        positioning = (repo_root / "tables" / "literature_positioning_map.md").read_text(encoding="utf-8")
        assert "Expected-cost CMDP optimization" in positioning
        assert "This empirical study" in positioning
        paper_positioning = (repo_root / "tables" / "paper_positioning_matrix.md").read_text(encoding="utf-8")
        assert "reporting-protocol contribution" in paper_positioning
        coverage = (repo_root / "tables" / "literature_metric_coverage.md").read_text(encoding="utf-8")
        assert "protocol reliability" in coverage
        assert "This empirical study" in coverage
        metric_family = (repo_root / "tables" / "metric_family_map.md").read_text(encoding="utf-8")
        assert "Temporal persistence" in metric_family
        quadrants = (repo_root / "tables" / "method_tradeoff_quadrants.md").read_text(encoding="utf-8")
        assert "higher return" in quadrants
        signature = (repo_root / "tables" / "method_safety_signature.md").read_text(encoding="utf-8")
        assert "short_max_run" in signature
        claim_flow = (repo_root / "tables" / "claim_flow.md").read_text(encoding="utf-8")
        assert "Claim boundary" in claim_flow
        key_numbers = (repo_root / "tables" / "key_numbers.md").read_text(encoding="utf-8")
        assert "Which method preserves reward?" in key_numbers
        disagreement = (repo_root / "tables" / "metric_disagreement_summary.md").read_text(encoding="utf-8")
        assert "max consecutive cost run" in disagreement
        reporting = (repo_root / "tables" / "statistical_reporting_checklist.md").read_text(encoding="utf-8")
        assert "Uncertainty context" in reporting
        assert "universal significance claim" in reporting
        protocol = (repo_root / "tables" / "reporting_protocol_upgrade.md").read_text(encoding="utf-8")
        assert "Episode-event reporting" in protocol
        cases = (repo_root / "tables" / "environment_case_studies.md").read_text(encoding="utf-8")
        assert "FOCOPS profile" in cases
        scorecard = (repo_root / "tables" / "env_method_scorecard.md").read_text(encoding="utf-8")
        assert "zero-violation gap" in scorecard
        assert "safe=" in scorecard
        coverage_matrix = (repo_root / "tables" / "protocol_coverage_matrix.md").read_text(encoding="utf-8")
        assert "Method coverage" in coverage_matrix
        assert "Not all Safe RL algorithms" in coverage_matrix
        assert not (repo_root / "experiments" / "results").exists()


def main_test():
    test_validate_rows_accepts_complete_matrix()
    test_read_metric_rows_uses_aggregate_archive_contract()
    test_build_artifacts_writes_paper_pack_without_results_dir()
    print("test_build_evidence_artifacts passed")


if __name__ == "__main__":
    main_test()
