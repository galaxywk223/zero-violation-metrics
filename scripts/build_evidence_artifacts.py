from __future__ import annotations

from argparse import ArgumentParser, Namespace, SUPPRESS
from pathlib import Path
import json
import math
import textwrap
import zipfile
from typing import Any

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]

EXPECTED_METHODS = ("PPO", "PPOLag", "FOCOPS", "CPO", "CPPOPID", "PPOSaute")
EXPECTED_ENVS = ("SafetyPointGoal1-v0", "SafetyPointButton1-v0", "SafetyCarGoal1-v0")
EXPECTED_SEEDS = (1, 2, 3)
EXPECTED_STEPS = 5_000_000

METRIC_COLUMNS = (
    "return",
    "mean_cost",
    "violation_rate",
    "safe_rate",
    "nonzero_cost_frequency",
    "p90_cost",
    "p95_cost",
    "max_cost",
    "conditional_unsafe_severity",
    "max_consecutive_cost_run",
)

HIGHER_IS_BETTER = {"return", "safe_rate"}
CORRELATION_METRICS = (
    "return",
    "mean_cost",
    "safe_rate",
    "nonzero_cost_frequency",
    "p95_cost",
    "max_consecutive_cost_run",
)
VARIABILITY_METRICS = (
    "return",
    "mean_cost",
    "safe_rate",
    "p95_cost",
    "max_consecutive_cost_run",
)
BOOTSTRAP_METRICS = ("return", "mean_cost", "safe_rate", "p95_cost")
BOOTSTRAP_SAMPLES = 10_000
BOOTSTRAP_SEED = 185
PROFILE_METRICS = (
    ("return", "return_mean", True),
    ("mean_cost", "mean_cost_mean", False),
    ("safe_rate", "safe_rate_mean", True),
    ("p95_cost", "p95_cost_mean", False),
    ("max_run", "max_consecutive_cost_run_mean", False),
)
METRIC_FAMILY_ROWS = (
    {
        "family": "Task performance",
        "metrics": "Return",
        "evaluation_question": "How much task performance remains after safety optimization?",
        "reported_evidence": "PPO has the highest return; CPPOPID and PPOSaute retain only about one fifth of PPO return.",
        "paper_role": "Prevents treating safer policies as automatically preferable when task performance collapses.",
    },
    {
        "family": "Expected-cost safety",
        "metrics": "Mean cost; violation rate",
        "evaluation_question": "How much safety cost is accumulated on average?",
        "reported_evidence": "Mean cost separates PPO from safety-aware baselines but does not determine zero-violation probability.",
        "paper_role": "Connects the study to the standard CMDP reporting convention.",
    },
    {
        "family": "Episode event safety",
        "metrics": "Safe rate; nonzero-cost episode frequency",
        "evaluation_question": "How often does an episode contain any violation?",
        "reported_evidence": "All evaluated methods retain nonzero-cost episodes; the best method-level safe rate remains below one.",
        "paper_role": "Defines the paper's central zero-violation reporting object.",
    },
    {
        "family": "Tail severity",
        "metrics": "p90 cost; p95 cost; max episode cost; conditional unsafe severity",
        "evaluation_question": "How severe are the remaining unsafe episodes?",
        "reported_evidence": "High safe rate can coexist with larger residual tails, especially across seed/environment slices.",
        "paper_role": "Separates rare-event magnitude from event frequency.",
    },
    {
        "family": "Temporal persistence",
        "metrics": "Maximum consecutive cost run",
        "evaluation_question": "Are violations isolated contacts or sustained unsafe runs?",
        "reported_evidence": "Safe-rate and max-run correlation is weak, motivating a separate persistence metric.",
        "paper_role": "Connects the empirical protocol to recent consecutive-violation safe-exploration metrics.",
    },
)
LITERATURE_POSITIONING_ROWS = (
    {
        "research_line": "Expected-cost CMDP optimization",
        "representative_examples": "CPO; PID Lagrangian; FOCOPS; CVPO; backward-value constraints",
        "primary_object": "Expected cumulative cost, budgeted cost, or state-specific cost constraints",
        "evaluation_gap": "Return-cost summaries can leave episode-level violation frequency and persistence implicit.",
        "study_role": "Audits whether mature expected-cost baselines also improve zero-violation metrics.",
        "x": 0.22,
        "y": 0.70,
        "color": "#4C78A8",
    },
    {
        "research_line": "Safe exploration and intervention",
        "representative_examples": "Safe exploration; EMCC; shielding; safety layers; recovery policies; CBF filters",
        "primary_object": "Unsafe states, actions, resets, or online intervention decisions",
        "evaluation_gap": "Mechanism papers often target direct prevention rather than benchmark reporting coverage.",
        "study_role": "Provides diagnostic readouts that intervention methods can be compared against.",
        "x": 0.76,
        "y": 0.72,
        "color": "#E45756",
    },
    {
        "research_line": "Risk, chance, and tail objectives",
        "representative_examples": "Percentile risk; chance-constrained MPC; adaptive chance safeguards; coherent risk; WCSAC",
        "primary_object": "Tail probability, worst-case behavior, or distributional safety criteria",
        "evaluation_gap": "Risk objectives motivate multiple safety summaries, but benchmark tables often collapse them.",
        "study_role": "Connects zero-violation frequency with tail severity and temporal persistence.",
        "x": 0.50,
        "y": 0.62,
        "color": "#F58518",
    },
    {
        "research_line": "Verified and hard-safety controllers",
        "representative_examples": "Verified Safe RL; robust CBF methods; CRABS; action projection",
        "primary_object": "Reachability, state constraints, or formally checked safe horizons",
        "evaluation_gap": "Formal guarantees and benchmark empirical metrics answer different questions.",
        "study_role": "Keeps empirical zero-violation reporting separate from guarantee claims.",
        "x": 0.86,
        "y": 0.46,
        "color": "#72B7B2",
    },
    {
        "research_line": "Benchmarks, datasets, and reliability",
        "representative_examples": "Safety-Gymnasium; OmniSafe; GUARD; SafeOR-Gym; Robust Gymnasium; D4RL; RL Unplugged",
        "primary_object": "Task coverage, algorithm coverage, reproducibility, intervals, and claim boundaries",
        "evaluation_gap": "Benchmark substrates need explicit metric panels and scoped interpretation.",
        "study_role": "Adds an episode-level zero-violation reporting layer to mature-baseline evaluation.",
        "x": 0.38,
        "y": 0.28,
        "color": "#54A24B",
    },
    {
        "research_line": "This empirical study",
        "representative_examples": "Six mature baselines; three Safety-Gymnasium environments; three seeds",
        "primary_object": "Return, mean cost, safe rate, nonzero frequency, tail severity, and run length",
        "evaluation_gap": "Focuses on evidence boundaries rather than a new optimizer or impossibility theorem.",
        "study_role": "Maps what the reported baseline matrix supports and what it does not support.",
        "x": 0.56,
        "y": 0.38,
        "color": "#B279A2",
    },
)

PAPER_POSITIONING_ROWS = (
    {
        "paper_family": "CMDP optimizers",
        "examples": "CPO; PPO-Lag; PID Lagrangian; FOCOPS; CVPO; Saute RL",
        "typical_evidence": "Return-cost curves, final return, expected cost, constraint satisfaction",
        "remaining_question": "Whether low average cost also means episode-level zero violation",
        "study_response": "Report safe rate, nonzero frequency, tail severity, and persistence beside return and mean cost.",
    },
    {
        "paper_family": "Intervention and hard-safety methods",
        "examples": "Shielding; safety layers; recovery policies; CBF filters; verified Safe RL",
        "typical_evidence": "Safe action sets, intervention frequency, reachability, formal guarantees",
        "remaining_question": "How benchmark baselines look before adding direct intervention mechanisms",
        "study_response": "Keep the baseline matrix as a reference map and avoid claiming formal safety guarantees.",
    },
    {
        "paper_family": "Risk and chance-constraint methods",
        "examples": "Percentile risk; coherent risk; distributional risk; WCSAC; adaptive chance safeguards",
        "typical_evidence": "Tail risk, violation probability bounds, worst-case or distributional objectives",
        "remaining_question": "Which empirical safety summaries should appear in a benchmark table",
        "study_response": "Use separate columns for expected cost, event frequency, tail magnitude, and consecutive-run behavior.",
    },
    {
        "paper_family": "RL benchmark and reliability papers",
        "examples": "Safety Gymnasium; OmniSafe; GUARD; D4RL; RL Unplugged; rliable",
        "typical_evidence": "Task coverage, fixed protocols, seeds, uncertainty summaries, reproducibility material",
        "remaining_question": "What the evaluation protocol can and cannot support",
        "study_response": "Make claim boundaries explicit and keep the contribution at the reporting-protocol layer.",
    },
)

LITERATURE_METRIC_COVERAGE_ROWS = (
    {
        "research_line": "Expected-cost Safe RL optimizers",
        "representative_papers": "CPO; PPO-Lag; PID Lagrangian; FOCOPS; Saute RL; CVPO",
        "expected_cost": 1.0,
        "episode_event": 0.25,
        "tail_severity": 0.25,
        "temporal_persistence": 0.0,
        "protocol_reliability": 0.5,
        "interpretation": "Core baselines optimize or report cost constraints, but episode-level event metrics are usually secondary.",
    },
    {
        "research_line": "Safety benchmark substrates",
        "representative_papers": "Safety Gym; Safety-Gymnasium; SafeLife; AI Safety Gridworlds",
        "expected_cost": 0.75,
        "episode_event": 0.5,
        "tail_severity": 0.25,
        "temporal_persistence": 0.0,
        "protocol_reliability": 1.0,
        "interpretation": "Benchmark papers make safety evaluation concrete, but the reported safety object varies by suite.",
    },
    {
        "research_line": "Implementation infrastructure",
        "representative_papers": "OmniSafe; Stable-Baselines3; CleanRL",
        "expected_cost": 0.5,
        "episode_event": 0.25,
        "tail_severity": 0.0,
        "temporal_persistence": 0.0,
        "protocol_reliability": 1.0,
        "interpretation": "Infrastructure papers justify controlled implementations and reproducible comparisons.",
    },
    {
        "research_line": "Risk and chance constraints",
        "representative_papers": "Coherent risk; chance-constrained safe RL; WCSAC; CVaR-PPO",
        "expected_cost": 0.5,
        "episode_event": 0.75,
        "tail_severity": 1.0,
        "temporal_persistence": 0.25,
        "protocol_reliability": 0.5,
        "interpretation": "Risk papers motivate safety summaries beyond the mean, especially probability and tail metrics.",
    },
    {
        "research_line": "Zero or bounded violation methods",
        "representative_papers": "Zero-constraint-violation primal-dual; Triple-Q; Safe Set Actor-Critic",
        "expected_cost": 0.5,
        "episode_event": 1.0,
        "tail_severity": 0.5,
        "temporal_persistence": 0.25,
        "protocol_reliability": 0.5,
        "interpretation": "Method-level zero-violation papers should be separated from reporting-protocol evidence.",
    },
    {
        "research_line": "RL evaluation reliability",
        "representative_papers": "RE-EVALUATE; rliable; seed-sensitivity studies; AdaStop",
        "expected_cost": 0.25,
        "episode_event": 0.25,
        "tail_severity": 0.25,
        "temporal_persistence": 0.25,
        "protocol_reliability": 1.0,
        "interpretation": "Evaluation papers motivate intervals, independent runs, and explicit claim boundaries.",
    },
    {
        "research_line": "This empirical study",
        "representative_papers": "Six mature baselines; three environments; three seeds",
        "expected_cost": 1.0,
        "episode_event": 1.0,
        "tail_severity": 1.0,
        "temporal_persistence": 1.0,
        "protocol_reliability": 0.75,
        "interpretation": "The study fills the reporting gap by placing all metric families in one baseline matrix.",
    },
)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build aggregate evidence artifacts from a sanitized metric table or archive.")
    parser.add_argument("--input-path", type=Path, help="JSON metric table or archive containing summaries/metric_table.json.")
    parser.add_argument("--archive-path", type=Path, help="Deprecated alias for --input-path.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--check-only", action="store_true")
    return parser


def resolve_path(path: Path, base: Path) -> Path:
    if path.is_absolute():
        return path
    return base / path


def resolve_args(args: Namespace) -> Namespace:
    if bool(args.input_path) == bool(args.archive_path):
        raise ValueError("provide exactly one of --input-path or --archive-path")
    args.input_path = args.input_path or args.archive_path
    args.repo_root = args.repo_root.resolve()
    args.input_path = resolve_path(args.input_path, args.repo_root).resolve()
    return args


def read_metric_rows(input_path: Path) -> list[dict[str, Any]]:
    if not input_path.exists():
        raise FileNotFoundError(f"Aggregate metric-table input not found: {input_path}")
    if input_path.suffix.lower() == ".json":
        payload = input_path.read_bytes()
    else:
        with zipfile.ZipFile(input_path) as archive:
            try:
                payload = archive.read("summaries/metric_table.json")
            except KeyError as exc:
                raise KeyError("Aggregate metric-table archive does not contain summaries/metric_table.json") from exc
    rows = json.loads(payload.decode("utf-8"))
    if not isinstance(rows, list):
        raise ValueError("metric-table input must contain a list of rows")
    return rows


def validate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    df = pd.DataFrame(rows)
    required_columns = {
        "run_id",
        "method",
        "env_id",
        "seed",
        "status",
        "evaluation_status",
        "training_executed",
        "requested_steps",
        "actual_steps",
        *METRIC_COLUMNS,
    }
    missing_columns = sorted(required_columns.difference(df.columns))
    if missing_columns:
        raise ValueError(f"missing required columns: {missing_columns}")

    methods = tuple(sorted(df["method"].dropna().unique()))
    envs = tuple(sorted(df["env_id"].dropna().unique()))
    seeds = tuple(sorted(int(seed) for seed in df["seed"].dropna().unique()))
    expected_methods = tuple(sorted(EXPECTED_METHODS))
    expected_envs = tuple(sorted(EXPECTED_ENVS))

    errors: list[str] = []
    if len(df) != 54:
        errors.append(f"expected 54 rows, got {len(df)}")
    if methods != expected_methods:
        errors.append(f"expected methods {expected_methods}, got {methods}")
    if envs != expected_envs:
        errors.append(f"expected envs {expected_envs}, got {envs}")
    if seeds != EXPECTED_SEEDS:
        errors.append(f"expected seeds {EXPECTED_SEEDS}, got {seeds}")
    if int((df["status"] == "completed").sum()) != 54:
        errors.append("not all runs have status=completed")
    if int((df["evaluation_status"] == "completed").sum()) != 54:
        errors.append("not all runs have evaluation_status=completed")
    if int((df["training_executed"] == True).sum()) != 54:  # noqa: E712
        errors.append("not all runs have training_executed=true")
    if not (df["requested_steps"].astype(float) == float(EXPECTED_STEPS)).all():
        errors.append("not all requested_steps equal 5,000,000")
    if not (df["actual_steps"].astype(float) == float(EXPECTED_STEPS)).all():
        errors.append("not all actual_steps equal 5,000,000")
    null_metrics = [
        metric
        for metric in METRIC_COLUMNS
        if df[metric].isna().any() or not pd.to_numeric(df[metric], errors="coerce").notna().all()
    ]
    if null_metrics:
        errors.append(f"metrics contain missing or nonnumeric values: {null_metrics}")
    if errors:
        raise ValueError("; ".join(errors))

    return {
        "rows": len(df),
        "methods": list(methods),
        "envs": list(envs),
        "seeds": list(seeds),
        "completed": int((df["status"] == "completed").sum()),
        "completed_evaluations": int((df["evaluation_status"] == "completed").sum()),
    }


def prepare_dataframe(rows: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df["method"] = pd.Categorical(df["method"], categories=EXPECTED_METHODS, ordered=True)
    df["env_id"] = pd.Categorical(df["env_id"], categories=EXPECTED_ENVS, ordered=True)
    for column in ("seed", "requested_steps", "actual_steps", *METRIC_COLUMNS):
        df[column] = pd.to_numeric(df[column])
    return df.sort_values(["env_id", "method", "seed"]).reset_index(drop=True)


def aggregate_overall(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("method", observed=False)
    out = grouped[list(METRIC_COLUMNS)].agg(["mean", "std"]).reset_index()
    out.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in out.columns
    ]
    out["run_count"] = grouped.size().to_numpy()
    return out


def aggregate_env_method(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["env_id", "method"], observed=False)
    out = grouped[list(METRIC_COLUMNS)].mean().reset_index()
    out["run_count"] = grouped.size().to_numpy()
    return out


def aggregate_seed_variability(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["env_id", "method"], observed=False)
    env_seed = grouped[list(VARIABILITY_METRICS)].std(ddof=1).reset_index()
    method_variability = (
        env_seed.groupby("method", observed=False)[list(VARIABILITY_METRICS)]
        .mean()
        .reset_index()
        .rename(columns={metric: f"{metric}_seed_std" for metric in VARIABILITY_METRICS})
    )
    method_variability["env_cells"] = (
        env_seed.groupby("method", observed=False).size().reset_index(name="env_cells")["env_cells"].to_numpy()
    )
    return method_variability


def compute_metric_correlation(df: pd.DataFrame) -> pd.DataFrame:
    return df[list(CORRELATION_METRICS)].corr(method="pearson")


def compute_bootstrap_ci(
    df: pd.DataFrame,
    metrics: tuple[str, ...] = BOOTSTRAP_METRICS,
    samples: int = BOOTSTRAP_SAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, Any]] = []
    for method, items in df.groupby("method", observed=False):
        method_label = str(method)
        for metric in metrics:
            values = items[metric].to_numpy(dtype=float)
            draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
            rows.append(
                {
                    "method": method_label,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "ci95_low": float(np.quantile(draws, 0.025)),
                    "ci95_high": float(np.quantile(draws, 0.975)),
                    "n": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def build_environment_best(env_method: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    specs = [
        ("highest_return", "return", True),
        ("highest_safe_rate", "safe_rate", True),
        ("lowest_mean_cost", "mean_cost", False),
        ("lowest_p95_cost", "p95_cost", False),
        ("shortest_max_run", "max_consecutive_cost_run", False),
    ]
    for env_id, items in env_method.groupby("env_id", observed=False):
        row: dict[str, Any] = {"env_id": str(env_id)}
        for label, metric, higher in specs:
            idx = items[metric].idxmax() if higher else items[metric].idxmin()
            best = items.loc[idx]
            row[label] = f"{best['method']} ({fmt_num(best[metric])})"
        rows.append(row)
    return pd.DataFrame(rows)


def build_method_rank_profile(overall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    for metric_label, column, higher in PROFILE_METRICS:
        ranks = view[column].rank(method="average", ascending=not higher)
        for method, value, rank in zip(view["method"], view[column], ranks, strict=False):
            rows.append(
                {
                    "method": method,
                    "metric": metric_label,
                    "value": float(value),
                    "rank": float(rank),
                }
            )
    return pd.DataFrame(rows)


def build_normalized_profiles(overall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    for metric_label, column, higher in PROFILE_METRICS:
        values = view[column].astype(float)
        span = float(values.max() - values.min())
        if span == 0.0:
            normalized = pd.Series([1.0] * len(view), index=view.index)
        elif higher:
            normalized = (values - values.min()) / span
        else:
            normalized = (values.max() - values) / span
        for method, value, score in zip(view["method"], values, normalized, strict=False):
            rows.append(
                {
                    "method": method,
                    "metric": metric_label,
                    "value": float(value),
                    "normalized_score": float(score),
                }
            )
    return pd.DataFrame(rows)


def build_relative_to_ppo_tradeoffs(overall: pd.DataFrame) -> pd.DataFrame:
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    ppo = row_for(view, "PPO")
    ppo_return = float(ppo["return_mean"])
    ppo_mean_cost = float(ppo["mean_cost_mean"])
    ppo_safe_rate = float(ppo["safe_rate_mean"])
    ppo_nonzero = float(ppo["nonzero_cost_frequency_mean"])
    ppo_p95 = float(ppo["p95_cost_mean"])

    rows: list[dict[str, Any]] = []
    for _, row in view.iterrows():
        method = str(row["method"])
        return_mean = float(row["return_mean"])
        mean_cost = float(row["mean_cost_mean"])
        safe_rate = float(row["safe_rate_mean"])
        nonzero = float(row["nonzero_cost_frequency_mean"])
        p95 = float(row["p95_cost_mean"])
        rows.append(
            {
                "method": method,
                "return_mean": return_mean,
                "return_retained_pct": 100.0 * return_mean / ppo_return,
                "return_delta": return_mean - ppo_return,
                "safe_rate": safe_rate,
                "safe_rate_gain": safe_rate - ppo_safe_rate,
                "unsafe_episode_reduction_pct": 100.0 * (ppo_nonzero - nonzero) / ppo_nonzero,
                "mean_cost_reduction_pct": 100.0 * (ppo_mean_cost - mean_cost) / ppo_mean_cost,
                "p95_cost_reduction_pct": 100.0 * (ppo_p95 - p95) / ppo_p95,
            }
        )
    return pd.DataFrame(rows)


def build_claim_boundary(
    overall: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    relative_tradeoffs: pd.DataFrame,
) -> pd.DataFrame:
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    ppo = row_for(view, "PPO")
    focops = row_for(view, "FOCOPS")
    cppopid = row_for(view, "CPPOPID")
    pposaute = row_for(view, "PPOSaute")
    max_safe_rate = float(view["safe_rate_mean"].max())
    return_safe_corr = float(metric_correlation.loc["return", "safe_rate"])
    mean_cost_safe_corr = float(metric_correlation.loc["mean_cost", "safe_rate"])
    p95_run_corr = float(metric_correlation.loc["p95_cost", "max_consecutive_cost_run"])
    focops_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "FOCOPS"].iloc[0]
    cppopid_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "CPPOPID"].iloc[0]
    pporsaute_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "PPOSaute"].iloc[0]
    safest_method = "CPPOPID" if float(cppopid["safe_rate_mean"]) >= float(pposaute["safe_rate_mean"]) else "PPOSaute"
    safest_return = float(cppopid["return_mean"]) if safest_method == "CPPOPID" else float(pposaute["return_mean"])
    safest_safe_rate = float(cppopid["safe_rate_mean"]) if safest_method == "CPPOPID" else float(pposaute["safe_rate_mean"])

    return pd.DataFrame(
        [
            {
                "claim": "Expected-cost and zero-violation metrics diverge.",
                "evidence": (
                    f"Return/safe-rate correlation is {fmt_num(return_safe_corr, 2)}; "
                    f"mean-cost/safe-rate correlation is {fmt_num(mean_cost_safe_corr, 2)}; "
                    f"p95/run-length correlation is {fmt_num(p95_run_corr, 2)}."
                ),
                "boundary": "Benchmark-scoped empirical evidence; not a theoretical separation theorem.",
                "not_supported": "Expected cost is useless or should be removed.",
            },
            {
                "claim": "No evaluated mature baseline reaches true zero violation.",
                "evidence": f"The best method-level safe rate is {fmt_num(max_safe_rate)}, below 1.000.",
                "boundary": "Six methods, three environments, three seeds, and one training budget.",
                "not_supported": "All Safe RL methods fail under all settings.",
            },
            {
                "claim": "PPO is reward-strong and safety-weak.",
                "evidence": (
                    f"PPO return is {fmt_num(ppo['return_mean'])}; safe rate is "
                    f"{fmt_num(ppo['safe_rate_mean'])}; nonzero-cost frequency is "
                    f"{fmt_num(ppo['nonzero_cost_frequency_mean'])}."
                ),
                "boundary": "PPO is a reward reference point, not a safety baseline.",
                "not_supported": "High return is acceptable when violation probability remains high.",
            },
            {
                "claim": "CPPOPID and PPOSaute improve safe rate with large return loss.",
                "evidence": (
                    f"{safest_method} safe rate is {fmt_num(safest_safe_rate)} with return "
                    f"{fmt_num(safest_return)}; CPPOPID and PPOSaute retain "
                    f"{fmt_num(cppopid_tradeoff['return_retained_pct'])}% and "
                    f"{fmt_num(pporsaute_tradeoff['return_retained_pct'])}% of PPO return."
                ),
                "boundary": "The result identifies a trade-off, not a cost-free safety improvement.",
                "not_supported": "The safest safe-rate method is automatically the best method.",
            },
            {
                "claim": "FOCOPS is the balanced comparator.",
                "evidence": (
                    f"FOCOPS return is {fmt_num(focops['return_mean'])}; safe rate is "
                    f"{fmt_num(focops['safe_rate_mean'])}; unsafe episodes are reduced by "
                    f"{fmt_num(focops_tradeoff['unsafe_episode_reduction_pct'])}% relative to PPO."
                ),
                "boundary": "FOCOPS remains below true zero violation.",
                "not_supported": "FOCOPS solves the zero-violation problem.",
            },
            {
                "claim": "A metric panel is the paper contribution.",
                "evidence": "Mean cost, safe rate, tail cost, run length, seed variability, and ranks expose different behavior.",
                "boundary": "The contribution is an evaluation protocol and evidence map, not a new algorithmic guarantee.",
                "not_supported": "The study reports a successful prototype zero-violation algorithm.",
            },
        ]
    )


def build_main_findings_summary(
    overall: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    relative_tradeoffs: pd.DataFrame,
) -> pd.DataFrame:
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    ppo = row_for(view, "PPO")
    focops = row_for(view, "FOCOPS")
    cppopid = row_for(view, "CPPOPID")
    pporsaute = row_for(view, "PPOSaute")
    focops_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "FOCOPS"].iloc[0]
    cppopid_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "CPPOPID"].iloc[0]
    pporsaute_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "PPOSaute"].iloc[0]
    mean_cost_safe_corr = float(metric_correlation.loc["mean_cost", "safe_rate"])
    safe_run_corr = float(metric_correlation.loc["safe_rate", "max_consecutive_cost_run"])
    p95_run_corr = float(metric_correlation.loc["p95_cost", "max_consecutive_cost_run"])

    return pd.DataFrame(
        [
            {
                "finding_id": "F1",
                "finding": "Reward-first optimization leaves frequent unsafe episodes.",
                "evidence": (
                    f"PPO has the highest return ({fmt_num(ppo['return_mean'])}) but safe rate "
                    f"{fmt_num(ppo['safe_rate_mean'])} and nonzero-cost frequency "
                    f"{fmt_num(ppo['nonzero_cost_frequency_mean'])}."
                ),
                "paper_use": "Defines the reward reference point and motivates event-level safety reporting.",
            },
            {
                "finding_id": "F2",
                "finding": "Safe-rate leaders are conservative and still not zero-violation.",
                "evidence": (
                    f"CPPOPID and PPOSaute safe rates are {fmt_num(cppopid['safe_rate_mean'])} and "
                    f"{fmt_num(pporsaute['safe_rate_mean'])}; return retained versus PPO is "
                    f"{fmt_num(cppopid_tradeoff['return_retained_pct'])}% and "
                    f"{fmt_num(pporsaute_tradeoff['return_retained_pct'])}%."
                ),
                "paper_use": "Separates safety improvement from cost-free deployment readiness.",
            },
            {
                "finding_id": "F3",
                "finding": "FOCOPS is the balanced comparator, not the zero-violation solution.",
                "evidence": (
                    f"FOCOPS return is {fmt_num(focops['return_mean'])}, safe rate is "
                    f"{fmt_num(focops['safe_rate_mean'])}, unsafe episodes reduced versus PPO by "
                    f"{fmt_num(focops_tradeoff['unsafe_episode_reduction_pct'])}%."
                ),
                "paper_use": "Sets the comparator that future zero-violation-oriented methods must beat.",
            },
            {
                "finding_id": "F4",
                "finding": "Safety metrics are coupled but not interchangeable.",
                "evidence": (
                    f"Mean-cost/safe-rate correlation is {fmt_num(mean_cost_safe_corr, 2)}; "
                    f"safe-rate/max-run correlation is {fmt_num(safe_run_corr, 2)}; "
                    f"p95/max-run correlation is {fmt_num(p95_run_corr, 2)}."
                ),
                "paper_use": "Supports the metric-panel recommendation instead of a single safety scalar.",
            },
        ]
    )


def build_literature_positioning_map() -> pd.DataFrame:
    return pd.DataFrame(LITERATURE_POSITIONING_ROWS)


def build_paper_positioning_matrix() -> pd.DataFrame:
    return pd.DataFrame(PAPER_POSITIONING_ROWS)


def build_literature_metric_coverage() -> pd.DataFrame:
    return pd.DataFrame(LITERATURE_METRIC_COVERAGE_ROWS)


def build_metric_family_map() -> pd.DataFrame:
    return pd.DataFrame(METRIC_FAMILY_ROWS)


def build_method_tradeoff_quadrants(overall: pd.DataFrame) -> pd.DataFrame:
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    return_threshold = float(view["return_mean"].median())
    safe_threshold = float(view["safe_rate_mean"].median())
    rows: list[dict[str, Any]] = []
    for _, row in view.iterrows():
        high_return = float(row["return_mean"]) >= return_threshold
        high_safe = float(row["safe_rate_mean"]) >= safe_threshold
        if high_return and high_safe:
            quadrant = "higher return / higher safe rate"
        elif high_return and not high_safe:
            quadrant = "higher return / lower safe rate"
        elif not high_return and high_safe:
            quadrant = "lower return / higher safe rate"
        else:
            quadrant = "lower return / lower safe rate"
        rows.append(
            {
                "method": str(row["method"]),
                "return_mean": float(row["return_mean"]),
                "safe_rate_mean": float(row["safe_rate_mean"]),
                "mean_cost_mean": float(row["mean_cost_mean"]),
                "p95_cost_mean": float(row["p95_cost_mean"]),
                "max_consecutive_cost_run_mean": float(row["max_consecutive_cost_run_mean"]),
                "return_threshold": return_threshold,
                "safe_rate_threshold": safe_threshold,
                "quadrant": quadrant,
            }
        )
    return pd.DataFrame(rows)


def build_method_safety_signature(overall: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    specs = (
        ("safe_rate", "safe_rate_mean", True),
        ("low_mean_cost", "mean_cost_mean", False),
        ("low_p95_cost", "p95_cost_mean", False),
        ("short_max_run", "max_consecutive_cost_run_mean", False),
        ("low_nonzero_frequency", "nonzero_cost_frequency_mean", False),
    )
    for label, column, higher in specs:
        values = view[column].astype(float)
        span = float(values.max() - values.min())
        if span == 0.0:
            scores = pd.Series([1.0] * len(view), index=view.index)
        elif higher:
            scores = (values - values.min()) / span
        else:
            scores = (values.max() - values) / span
        for method, value, score in zip(view["method"], values, scores, strict=False):
            rows.append(
                {
                    "method": method,
                    "safety_dimension": label,
                    "raw_value": float(value),
                    "normalized_score": float(score),
                }
            )
    return pd.DataFrame(rows)


def build_claim_flow_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "stage": "1",
                "component": "Evaluation object",
                "content": "Episode return and cost sequence define both reward and violation events.",
                "paper_role": "Establishes the measurement target.",
            },
            {
                "stage": "2",
                "component": "Metric families",
                "content": "Expected cost, safe rate, tail severity, and temporal persistence answer different questions.",
                "paper_role": "Prevents collapsing safety into one scalar.",
            },
            {
                "stage": "3",
                "component": "Baseline matrix",
                "content": "Six mature methods across three environments and three seeds expose the reward-safety landscape.",
                "paper_role": "Provides the empirical substrate.",
            },
            {
                "stage": "4",
                "component": "Supported claims",
                "content": "Metric non-equivalence, no true zero violation among evaluated baselines, and FOCOPS as balanced comparator.",
                "paper_role": "Defines what the paper can claim.",
            },
            {
                "stage": "5",
                "component": "Claim boundary",
                "content": "The study is not a universal failure theorem and not a new-method success result.",
                "paper_role": "Protects the conclusion from overstatement.",
            },
        ]
    )


def build_key_numbers(overall: pd.DataFrame, metric_correlation: pd.DataFrame, relative_tradeoffs: pd.DataFrame) -> pd.DataFrame:
    view = overall.copy()
    view["method"] = view["method"].astype(str)
    ppo = row_for(view, "PPO")
    focops = row_for(view, "FOCOPS")
    cppopid = row_for(view, "CPPOPID")
    pposaute = row_for(view, "PPOSaute")
    focops_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "FOCOPS"].iloc[0]
    cppopid_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "CPPOPID"].iloc[0]
    pposaute_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "PPOSaute"].iloc[0]
    best_safe = cppopid if float(cppopid["safe_rate_mean"]) >= float(pposaute["safe_rate_mean"]) else pposaute
    best_safe_tradeoff = (
        cppopid_tradeoff
        if str(best_safe["method"]) == "CPPOPID"
        else pposaute_tradeoff
    )
    return pd.DataFrame(
        [
            {
                "evaluation_question": "Which method preserves reward?",
                "answer": "PPO has the highest return.",
                "number": f"return={fmt_num(ppo['return_mean'])}, safe rate={fmt_num(ppo['safe_rate_mean'])}",
                "paper_role": "Defines the reward-only reference point.",
            },
            {
                "evaluation_question": "Which method has the highest safe rate?",
                "answer": f"{best_safe['method']} has the highest safe rate but remains below true zero violation.",
                "number": (
                    f"safe rate={fmt_num(best_safe['safe_rate_mean'])}, "
                    f"return retained={fmt_num(best_safe_tradeoff['return_retained_pct'])}%"
                ),
                "paper_role": "Separates safe-rate improvement from deployment-level zero violation.",
            },
            {
                "evaluation_question": "Which method is the balanced comparator?",
                "answer": "FOCOPS occupies the middle of the return-safety landscape.",
                "number": (
                    f"return={fmt_num(focops['return_mean'])}, safe rate={fmt_num(focops['safe_rate_mean'])}, "
                    f"unsafe reduction={fmt_num(focops_tradeoff['unsafe_episode_reduction_pct'])}%"
                ),
                "paper_role": "Defines the comparator for future zero-violation-oriented methods.",
            },
            {
                "evaluation_question": "Does mean cost determine zero-violation behavior?",
                "answer": "Mean cost and safe rate are strongly related but do not determine persistence.",
                "number": (
                    f"mean-cost/safe-rate r={fmt_num(metric_correlation.loc['mean_cost', 'safe_rate'], 2)}, "
                    f"safe-rate/max-run r={fmt_num(metric_correlation.loc['safe_rate', 'max_consecutive_cost_run'], 2)}"
                ),
                "paper_role": "Motivates reporting a metric panel rather than a single safety scalar.",
            },
        ]
    )


def build_metric_disagreement(overall: pd.DataFrame) -> pd.DataFrame:
    specs = [
        ("return", "return_mean", True),
        ("mean cost", "mean_cost_mean", False),
        ("safe rate", "safe_rate_mean", True),
        ("nonzero-cost frequency", "nonzero_cost_frequency_mean", False),
        ("p95 cost", "p95_cost_mean", False),
        ("max consecutive cost run", "max_consecutive_cost_run_mean", False),
    ]
    rows: list[dict[str, Any]] = []
    for label, column, higher in specs:
        sorted_view = overall.sort_values(column, ascending=not higher).reset_index(drop=True)
        best = sorted_view.iloc[0]
        worst = sorted_view.iloc[-1]
        rows.append(
            {
                "metric": label,
                "best_method": str(best["method"]),
                "best_value": float(best[column]),
                "worst_method": str(worst["method"]),
                "worst_value": float(worst[column]),
                "interpretation": (
                    "Higher value is better." if higher else "Lower value is better."
                ),
            }
        )
    return pd.DataFrame(rows)


def build_statistical_reporting_checklist() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "layer": "Protocol scope",
                "evidence_object": "6 methods x 3 environments x 3 seeds",
                "evaluation_question": "Which population of benchmark claims is covered?",
                "paper_use": "Define the evaluated scope before interpreting rankings.",
            },
            {
                "layer": "Independent runs",
                "evidence_object": "Environment-method-seed cells",
                "evaluation_question": "Are aggregate scores built from separate executions?",
                "paper_use": "Support descriptive comparisons while avoiding universal dominance claims.",
            },
            {
                "layer": "Metric families",
                "evidence_object": "Return, mean cost, safe rate, tails, max run",
                "evaluation_question": "Which safety object is being compared?",
                "paper_use": "Prevent expected-cost safety from being treated as zero-violation safety.",
            },
            {
                "layer": "Uncertainty context",
                "evidence_object": "Seed variability and bootstrap intervals",
                "evaluation_question": "How stable are method-level summaries?",
                "paper_use": "Report uncertainty without turning three seeds into strong significance claims.",
            },
            {
                "layer": "Claim boundary",
                "evidence_object": "Supported and unsupported claim table",
                "evaluation_question": "Which conclusions are justified by this evidence?",
                "paper_use": "Separate metric-reporting evidence from new-method or impossibility claims.",
            },
        ]
    )


def build_reporting_protocol_upgrade() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "reporting_layer": "Conventional return-cost summary",
                "reported_quantities": "Return; mean episode cost",
                "safety_question_answered": "How much cost is accumulated on average?",
                "question_left_open": "How often does any episode contain a violation?",
                "paper_use": "Defines the baseline reporting convention that the study extends.",
            },
            {
                "reporting_layer": "Episode-event reporting",
                "reported_quantities": "Safe rate; nonzero-cost episode frequency",
                "safety_question_answered": "How often is the episode cost exactly zero?",
                "question_left_open": "How severe are the residual unsafe episodes?",
                "paper_use": "Makes zero-violation behavior visible as an empirical target.",
            },
            {
                "reporting_layer": "Tail-severity reporting",
                "reported_quantities": "p90 cost; p95 cost; max episode cost; conditional unsafe severity",
                "safety_question_answered": "How large are the remaining unsafe episodes?",
                "question_left_open": "Are violations isolated or temporally persistent?",
                "paper_use": "Separates rare severe failures from frequent mild violations.",
            },
            {
                "reporting_layer": "Temporal-persistence reporting",
                "reported_quantities": "Maximum consecutive cost-positive run",
                "safety_question_answered": "Do violations persist across consecutive steps?",
                "question_left_open": "Whether a new optimizer or intervention mechanism can remove the residual gap.",
                "paper_use": "Prevents high safe rate from being read as stable violation-free behavior.",
            },
            {
                "reporting_layer": "Claim-boundary reporting",
                "reported_quantities": "Supported claims; unsupported readings; benchmark scope",
                "safety_question_answered": "Which conclusions are justified by the evidence matrix?",
                "question_left_open": "Generalization beyond the evaluated methods, tasks, seeds, and budget.",
                "paper_use": "Keeps the paper in the benchmark/evaluation genre and limits claims to the reported evidence.",
            },
        ]
    )


def build_environment_case_studies(env_method: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    takeaways = {
        "SafetyPointGoal1-v0": (
            "The easiest slice still separates reward leadership from safe-rate leadership; "
            "zero-violation reporting remains necessary even when costs are lower."
        ),
        "SafetyPointButton1-v0": (
            "The hardest slice exposes the central trade-off: PPO keeps return while safety-oriented "
            "methods raise safe rate with severe return loss."
        ),
        "SafetyCarGoal1-v0": (
            "The car-control slice highlights tail and persistence behavior; high safe rate does not "
            "eliminate the need to inspect p95 cost and maximum run length."
        ),
    }
    view = env_method.copy()
    view["env_id"] = view["env_id"].astype(str)
    view["method"] = view["method"].astype(str)
    for env_id in EXPECTED_ENVS:
        items = view.loc[view["env_id"] == env_id].copy()
        return_leader = items.loc[items["return"].idxmax()]
        safe_leader = items.loc[items["safe_rate"].idxmax()]
        mean_cost_leader = items.loc[items["mean_cost"].idxmin()]
        tail_leader = items.loc[items["p95_cost"].idxmin()]
        focops = items.loc[items["method"] == "FOCOPS"].iloc[0]
        rows.append(
            {
                "env_id": env_id,
                "return_leader": f"{return_leader['method']} ({fmt_num(return_leader['return'])})",
                "safe_rate_leader": f"{safe_leader['method']} ({fmt_num(safe_leader['safe_rate'])})",
                "mean_cost_leader": f"{mean_cost_leader['method']} ({fmt_num(mean_cost_leader['mean_cost'])})",
                "tail_leader": f"{tail_leader['method']} ({fmt_num(tail_leader['p95_cost'])})",
                "focops_profile": (
                    f"return={fmt_num(focops['return'])}; "
                    f"safe rate={fmt_num(focops['safe_rate'])}; "
                    f"p95={fmt_num(focops['p95_cost'])}; "
                    f"max run={fmt_num(focops['max_consecutive_cost_run'])}"
                ),
                "zero_violation_gap_best": 1.0 - float(safe_leader["safe_rate"]),
                "interpretation": takeaways[env_id],
            }
        )
    return pd.DataFrame(rows)


def build_env_method_scorecard(env_method: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    view = env_method.copy()
    view["env_id"] = view["env_id"].astype(str)
    view["method"] = view["method"].astype(str)
    for env_id in EXPECTED_ENVS:
        items = view.loc[view["env_id"] == env_id].copy()
        for method in EXPECTED_METHODS:
            row = items.loc[items["method"] == method].iloc[0]
            rows.append(
                {
                    "env_id": env_id,
                    "method": method,
                    "safe_rate": float(row["safe_rate"]),
                    "zero_violation_gap": 1.0 - float(row["safe_rate"]),
                    "return": float(row["return"]),
                    "mean_cost": float(row["mean_cost"]),
                    "p95_cost": float(row["p95_cost"]),
                    "max_consecutive_cost_run": float(row["max_consecutive_cost_run"]),
                    "paper_reading": (
                        f"safe={fmt_num(row['safe_rate'])}; "
                        f"return={fmt_num(row['return'])}; "
                        f"gap={fmt_num(1.0 - float(row['safe_rate']))}"
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_protocol_coverage_matrix(validation: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "coverage_axis": "Method coverage",
                "reported_coverage": f"{len(validation['methods'])} mature baselines",
                "evidence_object": "PPO, PPOLag, FOCOPS, CPO, CPPOPID, and PPOSaute",
                "claim_supported": "Baseline landscape and comparator selection",
                "claim_boundary": "Not all Safe RL algorithms or external implementations",
            },
            {
                "coverage_axis": "Task coverage",
                "reported_coverage": f"{len(validation['envs'])} Safety-Gymnasium tasks",
                "evidence_object": "PointGoal, PointButton, and CarGoal safety tasks",
                "claim_supported": "Environment-sliced trade-off behavior",
                "claim_boundary": "Not a full robotics, manipulation, or real-world benchmark",
            },
            {
                "coverage_axis": "Run coverage",
                "reported_coverage": f"{validation['rows']} completed method-environment-seed cells",
                "evidence_object": "Every method-environment-seed cell is present",
                "claim_supported": "Descriptive aggregate comparison under a fixed protocol",
                "claim_boundary": "Not enough for broad universal dominance claims",
            },
            {
                "coverage_axis": "Metric coverage",
                "reported_coverage": "Return, expectation, event frequency, tail, and persistence",
                "evidence_object": "Mean cost, safe rate, nonzero frequency, p95 cost, and max run",
                "claim_supported": "Expected-cost and zero-violation metrics are not interchangeable",
                "claim_boundary": "Not a formal proof of metric non-equivalence",
            },
            {
                "coverage_axis": "Artifact coverage",
                "reported_coverage": "Derived tables, figures, notes, and paper skeleton",
                "evidence_object": "Deterministic aggregate-evidence generator",
                "claim_supported": "Inspectability of the reported aggregate evidence pack",
                "claim_boundary": "No primary simulator traces, policy parameter files, or large result directories",
            },
        ]
    )


def mark_pareto_frontier(df: pd.DataFrame, x_col: str, y_col: str) -> pd.Series:
    flags: list[bool] = []
    points = df[[x_col, y_col]].astype(float)
    for idx, point in points.iterrows():
        dominated = False
        for other_idx, other in points.iterrows():
            if other_idx == idx:
                continue
            weakly_better = other[x_col] >= point[x_col] and other[y_col] >= point[y_col]
            strictly_better = other[x_col] > point[x_col] or other[y_col] > point[y_col]
            if weakly_better and strictly_better:
                dominated = True
                break
        flags.append(not dominated)
    return pd.Series(flags, index=df.index)


def build_rankings(env_method: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = ["return", "mean_cost", "safe_rate", "nonzero_cost_frequency", "p95_cost", "max_consecutive_cost_run"]
    safety_metrics = ["mean_cost", "safe_rate", "nonzero_cost_frequency", "p95_cost", "max_consecutive_cost_run"]
    return _rank_metrics(env_method, metrics), _rank_metrics(env_method, safety_metrics)


def _rank_metrics(env_method: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    rank_rows: list[dict[str, Any]] = []
    for env_id, items in env_method.groupby("env_id", observed=False):
        for metric in metrics:
            ascending = metric not in HIGHER_IS_BETTER
            ranks = items[metric].rank(method="average", ascending=ascending)
            for method, rank in zip(items["method"], ranks, strict=False):
                rank_rows.append({"env_id": env_id, "method": method, "metric": metric, "rank": float(rank)})
    rank_df = pd.DataFrame(rank_rows)
    return (
        rank_df.groupby("method", observed=False)["rank"]
        .mean()
        .reset_index(name="mean_rank")
        .sort_values("mean_rank")
        .reset_index(drop=True)
    )


def fmt_num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "missing"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return "missing"
    return f"{number:.{digits}f}"


def markdown_table(df: pd.DataFrame, columns: list[tuple[str, str]], digits: int = 3) -> str:
    header = "| " + " | ".join(label for label, _ in columns) + " |"
    divider = "| " + " | ".join("---" if idx == 0 else "---:" for idx, _ in enumerate(columns)) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        values = []
        for _, column in columns:
            value = row[column]
            if isinstance(value, (float, int)) and column not in {"run_count", "runs"}:
                values.append(fmt_num(value, digits))
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def wrap_text(value: Any, width: int) -> str:
    return "\n".join(textwrap.wrap(str(value), width=width, break_long_words=False))


def write_tables(
    repo_root: Path,
    overall: pd.DataFrame,
    env_method: pd.DataFrame,
    rankings: pd.DataFrame,
    safety_rankings: pd.DataFrame,
    seed_variability: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    environment_best: pd.DataFrame,
    method_rank_profile: pd.DataFrame,
    relative_tradeoffs: pd.DataFrame,
    claim_boundary: pd.DataFrame,
    main_findings_summary: pd.DataFrame,
    literature_positioning_map: pd.DataFrame,
    paper_positioning_matrix: pd.DataFrame,
    literature_metric_coverage: pd.DataFrame,
    metric_family_map: pd.DataFrame,
    method_tradeoff_quadrants: pd.DataFrame,
    method_safety_signature: pd.DataFrame,
    claim_flow_rows: pd.DataFrame,
    key_numbers: pd.DataFrame,
    metric_disagreement: pd.DataFrame,
    statistical_reporting_checklist: pd.DataFrame,
    reporting_protocol_upgrade: pd.DataFrame,
    environment_case_studies: pd.DataFrame,
    env_method_scorecard: pd.DataFrame,
    protocol_coverage_matrix: pd.DataFrame,
) -> None:
    tables_dir = repo_root / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    overall_view = pd.DataFrame(
        {
            "method": overall["method"].astype(str),
            "runs": overall["run_count"],
            "return_mean": overall["return_mean"],
            "return_std": overall["return_std"],
            "mean_cost_mean": overall["mean_cost_mean"],
            "safe_rate_mean": overall["safe_rate_mean"],
            "nonzero_cost_frequency_mean": overall["nonzero_cost_frequency_mean"],
            "p95_cost_mean": overall["p95_cost_mean"],
            "max_consecutive_cost_run_mean": overall["max_consecutive_cost_run_mean"],
        }
    )
    (tables_dir / "method_overall_metrics.md").write_text(
        "# Method-Level Overall Metrics\n\n"
        + markdown_table(
            overall_view,
            [
                ("method", "method"),
                ("runs", "runs"),
                ("return mean", "return_mean"),
                ("return std", "return_std"),
                ("mean cost", "mean_cost_mean"),
                ("safe rate", "safe_rate_mean"),
                ("nonzero freq", "nonzero_cost_frequency_mean"),
                ("p95 cost", "p95_cost_mean"),
                ("max run", "max_consecutive_cost_run_mean"),
            ],
        ),
        encoding="utf-8",
    )

    env_view = env_method.copy()
    env_view["env_id"] = env_view["env_id"].astype(str)
    env_view["method"] = env_view["method"].astype(str)
    (tables_dir / "env_method_metrics.md").write_text(
        "# Environment-Method Metrics\n\n"
        + markdown_table(
            env_view,
            [
                ("env", "env_id"),
                ("method", "method"),
                ("runs", "run_count"),
                ("return", "return"),
                ("mean cost", "mean_cost"),
                ("safe rate", "safe_rate"),
                ("nonzero freq", "nonzero_cost_frequency"),
                ("p95 cost", "p95_cost"),
                ("max run", "max_consecutive_cost_run"),
            ],
        ),
        encoding="utf-8",
    )

    rank_view = rankings.rename(columns={"mean_rank": "overall_mean_rank"}).merge(
        safety_rankings.rename(columns={"mean_rank": "safety_mean_rank"}),
        on="method",
        how="outer",
    )
    rank_view["method"] = rank_view["method"].astype(str)
    rank_view = rank_view.sort_values("overall_mean_rank")
    (tables_dir / "method_rankings.md").write_text(
        "# Method Rankings\n\n"
        + markdown_table(
            rank_view,
            [
                ("method", "method"),
                ("overall mean rank", "overall_mean_rank"),
                ("safety mean rank", "safety_mean_rank"),
            ],
        ),
        encoding="utf-8",
    )

    variability_view = seed_variability.copy()
    variability_view["method"] = variability_view["method"].astype(str)
    (tables_dir / "seed_variability.md").write_text(
        "# Seed Variability\n\n"
        "The table reports the mean within-environment standard deviation across three seeds. "
        "Each method has three environment cells.\n\n"
        + markdown_table(
            variability_view,
            [
                ("method", "method"),
                ("env cells", "env_cells"),
                ("return seed std", "return_seed_std"),
                ("mean cost seed std", "mean_cost_seed_std"),
                ("safe rate seed std", "safe_rate_seed_std"),
                ("p95 cost seed std", "p95_cost_seed_std"),
                ("max run seed std", "max_consecutive_cost_run_seed_std"),
            ],
        ),
        encoding="utf-8",
    )

    corr_view = metric_correlation.reset_index().rename(columns={"index": "metric"})
    (tables_dir / "metric_correlations.md").write_text(
        "# Metric Correlations\n\n"
        "Pearson correlations are computed over the 54-cell evaluation matrix.\n\n"
        + markdown_table(
            corr_view,
            [("metric", "metric"), *[(metric, metric) for metric in CORRELATION_METRICS]],
            digits=2,
        ),
        encoding="utf-8",
    )

    ci_view = bootstrap_ci.copy()
    (tables_dir / "bootstrap_ci.md").write_text(
        "# Bootstrap Confidence Intervals\n\n"
        f"The table reports method-level bootstrap 95% confidence intervals over the evaluation matrix. "
        f"Each method has `n=9` runs. Bootstrap samples: `{BOOTSTRAP_SAMPLES}`. Seed: `{BOOTSTRAP_SEED}`.\n\n"
        + markdown_table(
            ci_view,
            [
                ("method", "method"),
                ("metric", "metric"),
                ("n", "n"),
                ("mean", "mean"),
                ("ci95 low", "ci95_low"),
                ("ci95 high", "ci95_high"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "environment_wise_best_metrics.md").write_text(
        "# Environment-Wise Best Metrics\n\n"
        + markdown_table(
            environment_best,
            [
                ("env", "env_id"),
                ("highest return", "highest_return"),
                ("highest safe rate", "highest_safe_rate"),
                ("lowest mean cost", "lowest_mean_cost"),
                ("lowest p95 cost", "lowest_p95_cost"),
                ("shortest max run", "shortest_max_run"),
            ],
        ),
        encoding="utf-8",
    )

    rank_profile_view = method_rank_profile.copy().sort_values(["metric", "rank", "method"])
    (tables_dir / "method_metric_rank_profile.md").write_text(
        "# Method Metric Rank Profile\n\n"
        "Ranks are computed from method-level aggregate metrics. Rank 1 is best for the corresponding metric.\n\n"
        + markdown_table(
            rank_profile_view,
            [
                ("metric", "metric"),
                ("method", "method"),
                ("value", "value"),
                ("rank", "rank"),
            ],
        ),
        encoding="utf-8",
    )

    relative_view = relative_tradeoffs.copy()
    (tables_dir / "relative_to_ppo_tradeoffs.md").write_text(
        "# Relative-to-PPO Trade-Offs\n\n"
        "The table uses PPO as the reward-only reference point. Positive safety percentages indicate reductions "
        "relative to PPO's unsafe-episode frequency, mean cost, or p95 cost. Return retention reports the fraction "
        "of PPO return preserved by each method.\n\n"
        + markdown_table(
            relative_view,
            [
                ("method", "method"),
                ("return", "return_mean"),
                ("return retained %", "return_retained_pct"),
                ("return delta", "return_delta"),
                ("safe rate", "safe_rate"),
                ("safe-rate gain", "safe_rate_gain"),
                ("unsafe episode reduction %", "unsafe_episode_reduction_pct"),
                ("mean cost reduction %", "mean_cost_reduction_pct"),
                ("p95 cost reduction %", "p95_cost_reduction_pct"),
            ],
        ),
        encoding="utf-8",
    )

    claim_evidence = pd.DataFrame(
        [
            {
                "claim": "Expected-cost safety and episode-level zero-violation safety are not equivalent.",
                "evidence": "Metric correlations separate mean cost, safe rate, p95 cost, and maximum consecutive cost run.",
                "boundary": "The evidence is empirical and benchmark-scoped, not a theoretical impossibility result.",
            },
            {
                "claim": "Mature benchmark baselines do not reach true zero-violation behavior in the reported matrix.",
                "evidence": "The highest method-level safe rate remains below one, and every method retains nonzero-cost episodes.",
                "boundary": "The evidence does not cover every Safe RL algorithm, environment, or hyperparameter setting.",
            },
            {
                "claim": "Reward preservation and episode-level safety form a visible trade-off.",
                "evidence": "PPO has the highest return and lowest safe rate, while CPPOPID and PPOSaute have higher safe rates and lower returns.",
                "boundary": "The evidence supports a baseline trade-off map, not a universal Pareto frontier.",
            },
            {
                "claim": "FOCOPS is the most relevant balanced comparator for future zero-violation-oriented methods.",
                "evidence": "FOCOPS occupies the middle region across return, mean cost, safe rate, tail cost, and run-length metrics.",
                "boundary": "FOCOPS is not a zero-violation solution in this matrix.",
            },
            {
                "claim": "Safe RL evaluation should report a metric panel rather than only mean cost.",
                "evidence": "Environment slices, seed variability, bootstrap intervals, and rank profiles change across metrics.",
                "boundary": "The recommendation concerns evaluation reporting; it does not prescribe a new optimization objective.",
            },
        ]
    )
    (tables_dir / "claim_evidence_map.md").write_text(
        "# Claim-Evidence Map\n\n"
        + markdown_table(
            claim_evidence,
            [
                ("claim", "claim"),
                ("evidence", "evidence"),
                ("boundary", "boundary"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "claim_boundary.md").write_text(
        "# Claim Boundary\n\n"
        "The table translates the empirical result into claim-scoped boundaries. "
        "Each row pairs a paper-safe claim with the evidence, the admissible scope, and a common unsupported reading.\n\n"
        + markdown_table(
            claim_boundary,
            [
                ("claim", "claim"),
                ("evidence", "evidence"),
                ("boundary", "boundary"),
                ("not supported", "not_supported"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "main_findings_summary.md").write_text(
        "# Main Findings Summary\n\n"
        "The table condenses the reported baseline matrix into four main findings. "
        "Each finding is paired with quantitative evidence and its role in the paper narrative.\n\n"
        + markdown_table(
            main_findings_summary,
            [
                ("id", "finding_id"),
                ("finding", "finding"),
                ("evidence", "evidence"),
                ("paper use", "paper_use"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "literature_positioning_map.md").write_text(
        "# Literature Positioning Map\n\n"
        "The table maps the empirical study against adjacent Safe RL and RL evaluation lines. "
        "It is intended for paper positioning rather than for claiming algorithmic novelty.\n\n"
        + markdown_table(
            literature_positioning_map,
            [
                ("research line", "research_line"),
                ("representative examples", "representative_examples"),
                ("primary object", "primary_object"),
                ("evaluation gap", "evaluation_gap"),
                ("study role", "study_role"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "paper_positioning_matrix.md").write_text(
        "# Paper Positioning Matrix\n\n"
        "The table converts related-work families into paper-positioning decisions. "
        "It clarifies why the study is a reporting-protocol contribution rather than a new optimizer, "
        "a safety-filter method, or a formal guarantee paper.\n\n"
        + markdown_table(
            paper_positioning_matrix,
            [
                ("paper family", "paper_family"),
                ("examples", "examples"),
                ("typical evidence", "typical_evidence"),
                ("remaining question", "remaining_question"),
                ("Study response", "study_response"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "literature_metric_coverage.md").write_text(
        "# Literature Metric Coverage\n\n"
        "The table maps adjacent paper families to the metric families needed for episode-level zero-violation reporting. "
        "Coverage values are qualitative writing aids: 0 means mostly absent, 0.25 means indirect, 0.5 means partial, "
        "0.75 means prominent, and 1 means central. The table supports paper positioning and does not rank prior work.\n\n"
        + markdown_table(
            literature_metric_coverage,
            [
                ("research line", "research_line"),
                ("representative papers", "representative_papers"),
                ("expected cost", "expected_cost"),
                ("episode event", "episode_event"),
                ("tail severity", "tail_severity"),
                ("temporal persistence", "temporal_persistence"),
                ("protocol reliability", "protocol_reliability"),
                ("interpretation", "interpretation"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "metric_family_map.md").write_text(
        "# Metric Family Map\n\n"
        "The table groups the reported metrics by the safety question each family answers. "
        "It supports the paper's argument that return, expected cost, violation frequency, tail severity, "
        "and temporal persistence should be reported together.\n\n"
        + markdown_table(
            metric_family_map,
            [
                ("metric family", "family"),
                ("metrics", "metrics"),
                ("evaluation question", "evaluation_question"),
                ("Reported evidence", "reported_evidence"),
                ("paper role", "paper_role"),
            ],
        ),
        encoding="utf-8",
    )

    quadrant_view = method_tradeoff_quadrants.copy()
    (tables_dir / "method_tradeoff_quadrants.md").write_text(
        "# Method Trade-Off Quadrants\n\n"
        "The quadrant assignment uses the median method-level return and median method-level safe rate as reference lines. "
        "The table is descriptive and is not a universal Pareto claim.\n\n"
        + markdown_table(
            quadrant_view,
            [
                ("method", "method"),
                ("return", "return_mean"),
                ("safe rate", "safe_rate_mean"),
                ("mean cost", "mean_cost_mean"),
                ("p95 cost", "p95_cost_mean"),
                ("max run", "max_consecutive_cost_run_mean"),
                ("quadrant", "quadrant"),
            ],
        ),
        encoding="utf-8",
    )

    signature_view = method_safety_signature.copy().sort_values(["safety_dimension", "method"])
    (tables_dir / "method_safety_signature.md").write_text(
        "# Method Safety Signature\n\n"
        "The table normalizes safety-facing metrics so that higher scores are better. "
        "It supports visual comparison of frequency, severity, and persistence behavior.\n\n"
        + markdown_table(
            signature_view,
            [
                ("method", "method"),
                ("dimension", "safety_dimension"),
                ("raw value", "raw_value"),
                ("normalized score", "normalized_score"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "claim_flow.md").write_text(
        "# Claim Flow\n\n"
        "The table records the paper-level chain from evaluation object to claim boundary.\n\n"
        + markdown_table(
            claim_flow_rows,
            [
                ("stage", "stage"),
                ("component", "component"),
                ("content", "content"),
                ("paper role", "paper_role"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "key_numbers.md").write_text(
        "# Key Numbers for Main Text\n\n"
        "The table condenses the reported matrix into main-text quantities that can be cited in the main text.\n\n"
        + markdown_table(
            key_numbers,
            [
                ("evaluation question", "evaluation_question"),
                ("answer", "answer"),
                ("number", "number"),
                ("paper role", "paper_role"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "metric_disagreement_summary.md").write_text(
        "# Metric Disagreement Summary\n\n"
        "The table records which method is best and worst under each metric. The point is not a universal ranking, but the fact that the ranking target changes with the metric.\n\n"
        + markdown_table(
            metric_disagreement,
            [
                ("metric", "metric"),
                ("best method", "best_method"),
                ("best value", "best_value"),
                ("worst method", "worst_method"),
                ("worst value", "worst_value"),
                ("interpretation", "interpretation"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "statistical_reporting_checklist.md").write_text(
        "# Statistical Reporting Checklist\n\n"
        "The checklist turns statistical-comparison guidance into evidence requirements. "
        "It frames the aggregate matrix as scoped descriptive evidence rather than a universal significance claim.\n\n"
        + markdown_table(
            statistical_reporting_checklist,
            [
                ("layer", "layer"),
                ("evidence object", "evidence_object"),
                ("evaluation question", "evaluation_question"),
                ("paper use", "paper_use"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "reporting_protocol_upgrade.md").write_text(
        "# Reporting Protocol Upgrade\n\n"
        "The table records the paper's reporting move from conventional return-cost summaries to an episode-level "
        "zero-violation metric panel. It is a writing aid for keeping the manuscript in evaluation-paper form.\n\n"
        + markdown_table(
            reporting_protocol_upgrade,
            [
                ("reporting layer", "reporting_layer"),
                ("reported quantities", "reported_quantities"),
                ("answered question", "safety_question_answered"),
                ("question left open", "question_left_open"),
                ("paper use", "paper_use"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "environment_case_studies.md").write_text(
        "# Environment Case Studies\n\n"
        "The table converts environment slices into case studies. The purpose is to explain why "
        "environment-specific reporting is necessary instead of treating the aggregate average as the whole result.\n\n"
        + markdown_table(
            environment_case_studies,
            [
                ("environment", "env_id"),
                ("return leader", "return_leader"),
                ("safe-rate leader", "safe_rate_leader"),
                ("mean-cost leader", "mean_cost_leader"),
                ("tail leader", "tail_leader"),
                ("FOCOPS profile", "focops_profile"),
                ("best zero-violation gap", "zero_violation_gap_best"),
                ("interpretation", "interpretation"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "env_method_scorecard.md").write_text(
        "# Environment-Method Scorecard\n\n"
        "The scorecard reports each environment-method cell as a compact compact pair of zero-violation "
        "gap and return. It makes the full 3 x 6 evaluation matrix visible without collapsing the study into "
        "one aggregate leaderboard.\n\n"
        + markdown_table(
            env_method_scorecard,
            [
                ("environment", "env_id"),
                ("method", "method"),
                ("safe rate", "safe_rate"),
                ("zero-violation gap", "zero_violation_gap"),
                ("return", "return"),
                ("mean cost", "mean_cost"),
                ("p95 cost", "p95_cost"),
                ("max run", "max_consecutive_cost_run"),
                ("paper reading", "paper_reading"),
            ],
        ),
        encoding="utf-8",
    )

    (tables_dir / "protocol_coverage_matrix.md").write_text(
        "# Protocol Coverage Matrix\n\n"
        "The table states what the reported evidence covers and what it does not cover. It supports "
        "submission text and supporting-material documentation.\n\n"
        + markdown_table(
            protocol_coverage_matrix,
            [
                ("coverage axis", "coverage_axis"),
                ("reported coverage", "reported_coverage"),
                ("evidence object", "evidence_object"),
                ("claim supported", "claim_supported"),
                ("claim boundary", "claim_boundary"),
            ],
        ),
        encoding="utf-8",
    )


def set_figure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
        }
    )


def write_figures(
    repo_root: Path,
    env_method: pd.DataFrame,
    overall: pd.DataFrame,
    seed_variability: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    method_rank_profile: pd.DataFrame,
    normalized_profiles: pd.DataFrame,
    relative_tradeoffs: pd.DataFrame,
    claim_boundary: pd.DataFrame,
    main_findings_summary: pd.DataFrame,
    literature_positioning_map: pd.DataFrame,
    paper_positioning_matrix: pd.DataFrame,
    literature_metric_coverage: pd.DataFrame,
    metric_family_map: pd.DataFrame,
    method_tradeoff_quadrants: pd.DataFrame,
    method_safety_signature: pd.DataFrame,
    claim_flow_rows: pd.DataFrame,
    key_numbers: pd.DataFrame,
    metric_disagreement: pd.DataFrame,
    statistical_reporting_checklist: pd.DataFrame,
    reporting_protocol_upgrade: pd.DataFrame,
    environment_case_studies: pd.DataFrame,
    env_method_scorecard: pd.DataFrame,
    protocol_coverage_matrix: pd.DataFrame,
) -> None:
    figures_dir = repo_root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    set_figure_style()
    colors = {
        "PPO": "#4C78A8",
        "PPOLag": "#F58518",
        "FOCOPS": "#54A24B",
        "CPO": "#E45756",
        "CPPOPID": "#72B7B2",
        "PPOSaute": "#B279A2",
    }

    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    box_specs = [
        (0.05, 0.40, 0.18, 0.20, "Evaluation episodes\nreturn R, cost sequence c_t", "#E8EEF7"),
        (0.32, 0.40, 0.18, 0.20, "Episode cost\nC = sum_t c_t", "#F2F0E6"),
        (0.61, 0.72, 0.25, 0.16, "Expected-cost view\nmean cost, violation rate", "#EAF3EA"),
        (0.61, 0.50, 0.25, 0.16, "Zero-violation view\nsafe rate, nonzero frequency", "#F7EDEA"),
        (0.61, 0.28, 0.25, 0.16, "Tail and persistence view\np95 cost, max cost run", "#EFEAF6"),
        (0.18, 0.08, 0.64, 0.12, "Reported panel separates reward, average safety, event frequency, tail severity, and temporal persistence", "#F8F8F8"),
    ]
    for x0, y0, width, height, label, color in box_specs:
        patch = FancyBboxPatch(
            (x0, y0),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x0 + width / 2, y0 + height / 2, label, ha="center", va="center", fontsize=8.8)
    arrows = [
        ((0.23, 0.50), (0.32, 0.50)),
        ((0.50, 0.50), (0.61, 0.80)),
        ((0.50, 0.50), (0.61, 0.58)),
        ((0.50, 0.50), (0.61, 0.36)),
        ((0.735, 0.72), (0.50, 0.20)),
        ((0.735, 0.50), (0.50, 0.20)),
        ((0.735, 0.28), (0.50, 0.20)),
    ]
    for start, end in arrows:
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.0,
                color="#444444",
                shrinkA=4,
                shrinkB=4,
            )
        )
    ax.set_title("Episode-Level Zero-Violation Evaluation Protocol", fontsize=11, pad=8)
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_protocol_schematic.png")
    plt.close(fig)

    lit_plot = literature_positioning_map.copy()
    fig, ax = plt.subplots(figsize=(11.8, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Literature and Evaluation Positioning", fontsize=13, weight="bold", pad=10)
    ax.text(
        0.50,
        0.93,
        "The study adds an episode-level zero-violation reporting layer; it does not claim a new safety mechanism.",
        ha="center",
        va="center",
        fontsize=9,
        color="#333333",
    )
    center = lit_plot.loc[lit_plot["research_line"] == "This empirical study"].iloc[0]
    ax.add_patch(
        FancyBboxPatch(
            (0.27, 0.75),
            0.46,
            0.11,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            linewidth=1.5,
            edgecolor=center["color"],
            facecolor="#FFFFFF",
        )
    )
    ax.text(
        0.50,
        0.825,
        center["research_line"],
        ha="center",
        va="center",
        fontsize=11,
        weight="bold",
        color=center["color"],
    )
    ax.text(
        0.50,
        0.785,
        wrap_text(center["study_role"], 58),
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    ax.text(0.10, 0.685, "Adjacent research line", fontsize=9.2, weight="bold", color="#222222")
    ax.text(0.41, 0.685, "Primary safety object", fontsize=9.2, weight="bold", color="#222222")
    ax.text(0.73, 0.685, "Why a reporting layer is needed", fontsize=9.2, weight="bold", color="#222222")

    non_center = lit_plot[lit_plot["research_line"] != "This empirical study"].reset_index(drop=True)
    row_height = 0.112
    y_start = 0.62
    for idx, row in non_center.iterrows():
        y0 = y_start - idx * row_height
        ax.add_patch(
            FancyBboxPatch(
                (0.07, y0 - 0.058),
                0.86,
                0.087,
                boxstyle="round,pad=0.006,rounding_size=0.012",
                linewidth=0.9,
                edgecolor="#DDDDDD",
                facecolor="#FAFAFA" if idx % 2 == 0 else "#FFFFFF",
            )
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.083, y0 - 0.032),
                0.018,
                0.036,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                linewidth=0,
                facecolor=row["color"],
            )
        )
        ax.text(
            0.112,
            y0 - 0.014,
            wrap_text(row["research_line"], 27),
            ha="left",
            va="center",
            fontsize=8.4,
            weight="bold",
            color=row["color"],
        )
        ax.text(
            0.41,
            y0 - 0.014,
            wrap_text(row["primary_object"], 46),
            ha="center",
            va="center",
            fontsize=7.6,
            color="#333333",
        )
        ax.text(
            0.73,
            y0 - 0.014,
            wrap_text(row["evaluation_gap"], 50),
            ha="center",
            va="center",
            fontsize=7.6,
            color="#333333",
        )
    fig.tight_layout()
    fig.savefig(
        figures_dir / "literature_positioning_map.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12.2, 6.5))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Paper Positioning Matrix", fontsize=13, weight="bold", pad=10)
    ax.text(
        0.5,
        0.925,
        "The empirical contribution is a metric-reporting layer: aggregate evidence supports reporting claims, not a new safety mechanism.",
        ha="center",
        va="center",
        fontsize=8.9,
        color="#333333",
    )
    col_specs = [
        ("Paper family", 0.055, 0.17),
        ("Typical evidence", 0.25, 0.22),
        ("Question left open", 0.50, 0.22),
        ("Study response", 0.75, 0.22),
    ]
    for header, x0, width in col_specs:
        ax.text(x0 + width / 2, 0.84, header, ha="center", va="center", fontsize=8.8, weight="bold")
    row_height = 0.175
    y_top = 0.76
    palette = ["#4C78A8", "#E45756", "#F58518", "#54A24B"]
    for idx, row in paper_positioning_matrix.reset_index(drop=True).iterrows():
        y0 = y_top - idx * row_height
        face = "#FAFAFA" if idx % 2 == 0 else "#FFFFFF"
        ax.add_patch(
            FancyBboxPatch(
                (0.045, y0 - 0.077),
                0.91,
                0.135,
                boxstyle="round,pad=0.008,rounding_size=0.012",
                linewidth=0.8,
                edgecolor="#DDDDDD",
                facecolor=face,
            )
        )
        color = palette[idx % len(palette)]
        ax.add_patch(
            FancyBboxPatch(
                (0.058, y0 - 0.043),
                0.018,
                0.086,
                boxstyle="round,pad=0.002,rounding_size=0.006",
                linewidth=0,
                facecolor=color,
            )
        )
        ax.text(0.085, y0 + 0.018, wrap_text(row["paper_family"], 20), ha="left", va="center", fontsize=8.1, weight="bold", color=color)
        ax.text(0.085, y0 - 0.040, wrap_text(row["examples"], 28), ha="left", va="center", fontsize=6.9, color="#333333")
        ax.text(0.36, y0, wrap_text(row["typical_evidence"], 34), ha="center", va="center", fontsize=7.2, color="#333333")
        ax.text(0.61, y0, wrap_text(row["remaining_question"], 34), ha="center", va="center", fontsize=7.2, color="#333333")
        ax.text(0.86, y0, wrap_text(row["study_response"], 36), ha="center", va="center", fontsize=7.2, color="#333333")
    fig.tight_layout()
    fig.savefig(figures_dir / "paper_positioning_matrix.png", dpi=240, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    coverage_cols = [
        ("expected_cost", "Expected\ncost"),
        ("episode_event", "Episode\nevent"),
        ("tail_severity", "Tail\nseverity"),
        ("temporal_persistence", "Temporal\npersistence"),
        ("protocol_reliability", "Protocol\nreliability"),
    ]
    coverage_plot = literature_metric_coverage.copy()
    coverage_values = coverage_plot[[col for col, _ in coverage_cols]].astype(float).to_numpy()
    fig, ax = plt.subplots(figsize=(11.4, 6.5))
    image = ax.imshow(coverage_values, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(coverage_cols)))
    ax.set_xticklabels([label for _, label in coverage_cols], fontsize=8.2)
    ax.set_yticks(np.arange(len(coverage_plot)))
    ax.set_yticklabels(coverage_plot["research_line"], fontsize=8.0)
    ax.tick_params(axis="both", length=0)
    for i in range(coverage_values.shape[0]):
        for j in range(coverage_values.shape[1]):
            value = coverage_values[i, j]
            ax.text(
                j,
                i,
                fmt_num(value, digits=2),
                ha="center",
                va="center",
                fontsize=7.2,
                color="#111111" if value < 0.65 else "#FFFFFF",
                weight="bold" if value >= 0.75 else "normal",
            )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Literature-to-Metric Coverage Map", fontsize=13, weight="bold", pad=12)
    ax.set_xlabel("Metric or evaluation dimension", fontsize=9)
    ax.set_ylabel("Adjacent research line", fontsize=9)
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("qualitative coverage", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    caption = (
        "The map positions this study as a metric-completion layer: adjacent papers cover optimizers, benchmarks, "
        "risk criteria, implementation infrastructure, or statistical reliability, while the reported matrix places "
        "expected cost, zero-violation events, tails, and persistence in one paper table."
    )
    fig.text(0.50, 0.02, wrap_text(caption, 135), ha="center", va="bottom", fontsize=8.0, color="#333333")
    fig.tight_layout(rect=(0.03, 0.07, 1.0, 1.0))
    fig.savefig(
        figures_dir / "literature_metric_coverage.png",
        dpi=240,
        bbox_inches="tight",
        pad_inches=0.08,
    )
    plt.close(fig)

    overall_plot = overall.copy()
    overall_plot["method"] = overall_plot["method"].astype(str)

    fig, axes = plt.subplots(2, 2, figsize=(10.4, 7.0))
    x = np.arange(len(EXPECTED_METHODS))
    returns = [
        float(overall_plot.loc[overall_plot["method"] == method, "return_mean"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    safe_rates = [
        float(overall_plot.loc[overall_plot["method"] == method, "safe_rate_mean"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    mean_costs = [
        float(overall_plot.loc[overall_plot["method"] == method, "mean_cost_mean"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    max_runs = [
        float(overall_plot.loc[overall_plot["method"] == method, "max_consecutive_cost_run_mean"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    axes[0, 0].bar(x, returns, color=[colors[m] for m in EXPECTED_METHODS])
    axes[0, 0].set_title("Task Return")
    axes[0, 0].set_ylabel("Higher is better")
    axes[0, 1].bar(x, safe_rates, color=[colors[m] for m in EXPECTED_METHODS])
    axes[0, 1].axhline(1.0, color="#333333", linestyle="--", linewidth=1.0)
    axes[0, 1].set_ylim(0.0, 1.05)
    axes[0, 1].set_title("Episode-Level Safe Rate")
    axes[0, 1].set_ylabel("Higher is better")
    axes[1, 0].bar(x, mean_costs, color=[colors[m] for m in EXPECTED_METHODS])
    axes[1, 0].set_title("Expected-Cost Summary")
    axes[1, 0].set_ylabel("Lower is better")
    axes[1, 1].bar(x, max_runs, color=[colors[m] for m in EXPECTED_METHODS])
    axes[1, 1].set_title("Temporal Persistence")
    axes[1, 1].set_ylabel("Lower is better")
    for axis in axes.flat:
        axis.set_xticks(x)
        axis.set_xticklabels(EXPECTED_METHODS, rotation=30, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Claim-Aligned Main Evidence Across Metric Families", fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "claim_aligned_main_evidence.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    y = np.arange(len(metric_disagreement))
    best_labels = [
        f"{row.best_method}\n{fmt_num(row.best_value)}"
        for row in metric_disagreement.itertuples(index=False)
    ]
    worst_labels = [
        f"{row.worst_method}\n{fmt_num(row.worst_value)}"
        for row in metric_disagreement.itertuples(index=False)
    ]
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(metric_disagreement) - 0.4)
    ax.set_yticks(y)
    ax.set_yticklabels(metric_disagreement["metric"])
    ax.set_xticks([0.18, 0.82])
    ax.set_xticklabels(["Best under metric", "Worst under metric"])
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.15)
    for idx, (best, worst) in enumerate(zip(best_labels, worst_labels, strict=False)):
        ax.plot([0.18, 0.82], [idx, idx], color="#CCCCCC", linewidth=1.0)
        ax.scatter(0.18, idx, s=120, color="#54A24B", zorder=3)
        ax.scatter(0.82, idx, s=120, color="#E45756", zorder=3)
        ax.text(0.23, idx, best, ha="left", va="center", fontsize=8.2)
        ax.text(0.77, idx, worst, ha="right", va="center", fontsize=8.2)
    ax.set_title("Metric-Dependent Method Ordering")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_disagreement_summary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    rows = [
        (
            "Expected-cost view",
            "How much cost is accumulated on average?",
            "mean cost; violation rate",
            "#54A24B",
        ),
        (
            "Zero-violation view",
            "How often does an episode contain no violation?",
            "safe rate; nonzero-cost frequency",
            "#E45756",
        ),
        (
            "Tail and persistence view",
            "How severe and sustained are residual violations?",
            "p95 cost; max consecutive cost run",
            "#B279A2",
        ),
    ]
    ax.text(0.5, 0.93, "Expected Cost Does Not Specify Episode-Level Zero Violation", ha="center", va="center", fontsize=12, weight="bold")
    ax.text(0.5, 0.86, "The same episode-cost sequence supports multiple safety questions; a single scalar leaves the others under-specified.", ha="center", va="center", fontsize=8.8, color="#333333")
    for idx, (title, question, metrics, color) in enumerate(rows):
        y0 = 0.62 - idx * 0.22
        ax.add_patch(
            FancyBboxPatch(
                (0.08, y0 - 0.07),
                0.84,
                0.13,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.2,
                edgecolor=color,
                facecolor="#FFFFFF",
            )
        )
        ax.text(0.13, y0 + 0.025, title, ha="left", va="center", fontsize=9.2, weight="bold", color=color)
        ax.text(0.13, y0 - 0.025, question, ha="left", va="center", fontsize=8.2, color="#333333")
        ax.text(0.88, y0, metrics, ha="right", va="center", fontsize=8.0, color="#333333")
    fig.tight_layout()
    fig.savefig(figures_dir / "expected_cost_zero_violation_separation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.8, 5.2))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(
        0.5,
        0.93,
        "Statistical Reporting Ladder for the Evaluation Matrix",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    ax.text(
        0.5,
        0.86,
        "The evidence supports scoped empirical reporting, not universal method dominance or a new-method claim.",
        ha="center",
        va="center",
        fontsize=8.7,
        color="#333333",
    )
    y_positions = np.linspace(0.72, 0.18, len(statistical_reporting_checklist))
    palette = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756"]
    for idx, (_, row) in enumerate(statistical_reporting_checklist.iterrows()):
        y0 = float(y_positions[idx])
        color = palette[idx % len(palette)]
        ax.add_patch(
            FancyBboxPatch(
                (0.055, y0 - 0.055),
                0.89,
                0.094,
                boxstyle="round,pad=0.010,rounding_size=0.016",
                linewidth=1.0,
                edgecolor=color,
                facecolor="#FFFFFF",
            )
        )
        ax.text(0.075, y0 + 0.016, wrap_text(row["layer"], 20), ha="left", va="center", fontsize=8.7, weight="bold", color=color)
        ax.text(0.285, y0 + 0.016, wrap_text(row["evidence_object"], 26), ha="left", va="center", fontsize=7.6, color="#222222")
        ax.text(0.535, y0 + 0.016, wrap_text(row["evaluation_question"], 32), ha="left", va="center", fontsize=7.4, color="#333333")
        ax.text(0.535, y0 - 0.025, wrap_text(row["paper_use"], 48), ha="left", va="center", fontsize=7.2, color="#555555")
        if idx < len(statistical_reporting_checklist) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (0.5, y0 - 0.061),
                    (0.5, float(y_positions[idx + 1]) + 0.051),
                    arrowstyle="-|>",
                    mutation_scale=10,
                    linewidth=0.9,
                    color="#999999",
                )
            )
    fig.tight_layout()
    fig.savefig(figures_dir / "statistical_reporting_ladder.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    ppo_row = row_for(overall_plot, "PPO")
    focops_row = row_for(overall_plot, "FOCOPS")
    cppopid_row = row_for(overall_plot, "CPPOPID")
    pposaute_row = row_for(overall_plot, "PPOSaute")
    best_safe_rate = max(float(cppopid_row["safe_rate_mean"]), float(pposaute_row["safe_rate_mean"]))
    best_safe_method = "CPPOPID" if float(cppopid_row["safe_rate_mean"]) >= float(pposaute_row["safe_rate_mean"]) else "PPOSaute"
    max_safe_rate = float(overall_plot["safe_rate_mean"].max())
    corr_cost_safe = float(metric_correlation.loc["mean_cost", "safe_rate"])
    corr_safe_run = float(metric_correlation.loc["safe_rate", "max_consecutive_cost_run"])
    fig, axes = plt.subplots(2, 2, figsize=(10.2, 6.4))
    panels = [
        (
            axes[0, 0],
            "Reward-Safety Trade-off",
            f"PPO has the highest return\nreturn={fmt_num(ppo_row['return_mean'])}, safe rate={fmt_num(ppo_row['safe_rate_mean'])}",
            colors["PPO"],
        ),
        (
            axes[0, 1],
            "Conservative Safety Cost",
            f"{best_safe_method} has the highest safe rate\nsafe rate={fmt_num(best_safe_rate)}, below true zero violation",
            colors[best_safe_method],
        ),
        (
            axes[1, 0],
            "Balanced Comparator",
            f"FOCOPS sits in the middle\nreturn={fmt_num(focops_row['return_mean'])}, safe rate={fmt_num(focops_row['safe_rate_mean'])}",
            colors["FOCOPS"],
        ),
        (
            axes[1, 1],
            "Metric Non-Equivalence",
            f"mean cost vs safe rate r={fmt_num(corr_cost_safe, 2)}\nsafe rate vs max run r={fmt_num(corr_safe_run, 2)}",
            "#6F6F6F",
        ),
    ]
    for axis, title, body, color in panels:
        axis.set_axis_off()
        patch = FancyBboxPatch(
            (0.04, 0.12),
            0.92,
            0.76,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.2,
            edgecolor=color,
            facecolor="#FFFFFF",
        )
        axis.add_patch(patch)
        axis.text(0.5, 0.65, title, ha="center", va="center", fontsize=11, weight="bold", color=color)
        axis.text(0.5, 0.40, body, ha="center", va="center", fontsize=9)
    fig.suptitle(
        f"Core Takeaways from the Reported Baseline Matrix (max safe rate={fmt_num(max_safe_rate)})",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "core_takeaway_panel.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, row in overall_plot.iterrows():
        method = row["method"]
        ax.scatter(row["safe_rate_mean"], row["return_mean"], s=72, color=colors[method], label=method)
        ax.annotate(method, (row["safe_rate_mean"], row["return_mean"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Safe rate")
    ax.set_ylabel("Return")
    ax.set_title("Return vs Episode-Level Safe Rate")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "return_vs_safe_rate.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, row in overall_plot.iterrows():
        method = row["method"]
        ax.scatter(row["nonzero_cost_frequency_mean"], row["mean_cost_mean"], s=72, color=colors[method])
        ax.annotate(method, (row["nonzero_cost_frequency_mean"], row["mean_cost_mean"]), xytext=(5, 4), textcoords="offset points")
    ax.set_xlabel("Nonzero-cost episode frequency")
    ax.set_ylabel("Mean cost")
    ax.set_title("Mean Cost vs Nonzero-Cost Frequency")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "mean_cost_vs_nonzero_frequency.png")
    plt.close(fig)

    x = range(len(EXPECTED_METHODS))
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    p95_vals = [float(overall_plot.loc[overall_plot["method"] == method, "p95_cost_mean"].iloc[0]) for method in EXPECTED_METHODS]
    run_vals = [
        float(overall_plot.loc[overall_plot["method"] == method, "max_consecutive_cost_run_mean"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    axes[0].bar(x, p95_vals, color=[colors[m] for m in EXPECTED_METHODS])
    axes[0].set_title("p95 Cost")
    axes[0].set_ylabel("Cost")
    axes[1].bar(x, run_vals, color=[colors[m] for m in EXPECTED_METHODS])
    axes[1].set_title("Max Consecutive Cost Run")
    axes[1].set_ylabel("Steps")
    for axis in axes:
        axis.set_xticks(list(x))
        axis.set_xticklabels(EXPECTED_METHODS, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Tail and Run-Length Safety Metrics")
    fig.tight_layout()
    fig.savefig(figures_dir / "tail_and_run_metrics.png")
    plt.close(fig)

    heat = env_method.pivot(index="env_id", columns="method", values="safe_rate").reindex(EXPECTED_ENVS)
    heat = heat[list(EXPECTED_METHODS)]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    image = ax.imshow(heat.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(EXPECTED_METHODS)))
    ax.set_xticklabels(EXPECTED_METHODS, rotation=30, ha="right")
    ax.set_yticks(range(len(EXPECTED_ENVS)))
    ax.set_yticklabels(EXPECTED_ENVS)
    for i, env_id in enumerate(EXPECTED_ENVS):
        for j, method in enumerate(EXPECTED_METHODS):
            ax.text(j, i, fmt_num(heat.loc[env_id, method], 2), ha="center", va="center", color="black")
    ax.set_title("Safe Rate by Environment and Method")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Safe rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "env_method_heatmap.png")
    plt.close(fig)

    safe_pivot = env_method_scorecard.pivot(index="env_id", columns="method", values="safe_rate").reindex(EXPECTED_ENVS)
    return_pivot = env_method_scorecard.pivot(index="env_id", columns="method", values="return").reindex(EXPECTED_ENVS)
    gap_pivot = env_method_scorecard.pivot(index="env_id", columns="method", values="zero_violation_gap").reindex(EXPECTED_ENVS)
    safe_pivot = safe_pivot[list(EXPECTED_METHODS)]
    return_pivot = return_pivot[list(EXPECTED_METHODS)]
    gap_pivot = gap_pivot[list(EXPECTED_METHODS)]
    fig, ax = plt.subplots(figsize=(10.4, 4.7))
    image = ax.imshow(safe_pivot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(EXPECTED_METHODS)))
    ax.set_xticklabels(EXPECTED_METHODS, rotation=25, ha="right")
    ax.set_yticks(range(len(EXPECTED_ENVS)))
    ax.set_yticklabels(EXPECTED_ENVS)
    ax.tick_params(length=0)
    for i, env_id in enumerate(EXPECTED_ENVS):
        for j, method in enumerate(EXPECTED_METHODS):
            safe_value = float(safe_pivot.loc[env_id, method])
            return_value = float(return_pivot.loc[env_id, method])
            gap_value = float(gap_pivot.loc[env_id, method])
            text_color = "white" if safe_value > 0.62 else "#111111"
            ax.text(j, i - 0.14, f"safe {safe_value:.2f}", ha="center", va="center", fontsize=7.4, color=text_color, weight="bold")
            ax.text(j, i + 0.08, f"ret {return_value:.2f}", ha="center", va="center", fontsize=7.0, color=text_color)
            ax.text(j, i + 0.27, f"gap {gap_value:.2f}", ha="center", va="center", fontsize=6.6, color=text_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title("Environment-Method Zero-Violation Scorecard", fontsize=12, weight="bold")
    ax.set_xlabel("Method")
    ax.set_ylabel("Environment")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("safe rate", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    caption = (
        "Each cell shows safe rate, return, and residual zero-violation gap. "
        "The full matrix makes task-specific reward-safety trade-offs visible without turning the paper into a single leaderboard."
    )
    fig.text(0.50, 0.01, wrap_text(caption, 130), ha="center", va="bottom", fontsize=7.8, color="#333333")
    fig.tight_layout(rect=(0.02, 0.06, 1.0, 1.0))
    fig.savefig(figures_dir / "env_method_scorecard.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    pareto_plot = overall_plot.copy()
    pareto_plot["pareto"] = mark_pareto_frontier(pareto_plot, "safe_rate_mean", "return_mean")
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for _, row in pareto_plot.iterrows():
        method = row["method"]
        ax.scatter(
            row["safe_rate_mean"],
            row["return_mean"],
            s=92 if row["pareto"] else 64,
            color=colors[method],
            edgecolor="black" if row["pareto"] else "white",
            linewidth=1.1 if row["pareto"] else 0.6,
            alpha=0.95,
        )
        ax.annotate(method, (row["safe_rate_mean"], row["return_mean"]), xytext=(5, 4), textcoords="offset points")
    frontier = pareto_plot.loc[pareto_plot["pareto"]].sort_values("safe_rate_mean")
    ax.plot(frontier["safe_rate_mean"], frontier["return_mean"], linestyle="--", color="#333333", alpha=0.55)
    ax.set_xlabel("Safe rate")
    ax.set_ylabel("Return")
    ax.set_title("Return-Safe-Rate Pareto Frontier")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "pareto_frontier.png")
    plt.close(fig)

    variability_plot = seed_variability.copy()
    variability_plot["method"] = variability_plot["method"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    x = range(len(EXPECTED_METHODS))
    return_std = [
        float(variability_plot.loc[variability_plot["method"] == method, "return_seed_std"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    safe_std = [
        float(variability_plot.loc[variability_plot["method"] == method, "safe_rate_seed_std"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    axes[0].bar(x, return_std, color=[colors[m] for m in EXPECTED_METHODS])
    axes[0].set_title("Return Seed Std")
    axes[0].set_ylabel("Std across seeds")
    axes[1].bar(x, safe_std, color=[colors[m] for m in EXPECTED_METHODS])
    axes[1].set_title("Safe-Rate Seed Std")
    axes[1].set_ylabel("Std across seeds")
    for axis in axes:
        axis.set_xticks(list(x))
        axis.set_xticklabels(EXPECTED_METHODS, rotation=35, ha="right")
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Within-Environment Seed Variability")
    fig.tight_layout()
    fig.savefig(figures_dir / "seed_variability.png")
    plt.close(fig)

    labels = list(CORRELATION_METRICS)
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    image = ax.imshow(metric_correlation.to_numpy(dtype=float), cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    for i, row_metric in enumerate(labels):
        for j, col_metric in enumerate(labels):
            value = metric_correlation.loc[row_metric, col_metric]
            ax.text(j, i, fmt_num(value, 2), ha="center", va="center", color="black")
    ax.set_title("Metric Correlation Heatmap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Pearson correlation")
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_correlation_heatmap.png")
    plt.close(fig)

    ci_plot = bootstrap_ci.copy()
    ci_plot["method"] = ci_plot["method"].astype(str)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.4), sharey=True)
    y_pos = np.arange(len(EXPECTED_METHODS))
    for axis, metric, title, xlabel in [
        (axes[0], "return", "Return", "Return"),
        (axes[1], "safe_rate", "Safe Rate", "Safe rate"),
    ]:
        view = ci_plot.loc[ci_plot["metric"] == metric].set_index("method").reindex(EXPECTED_METHODS)
        means = view["mean"].to_numpy(dtype=float)
        lows = view["ci95_low"].to_numpy(dtype=float)
        highs = view["ci95_high"].to_numpy(dtype=float)
        xerr = np.vstack([means - lows, highs - means])
        axis.errorbar(
            means,
            y_pos,
            xerr=xerr,
            fmt="o",
            color="#333333",
            ecolor="#777777",
            elinewidth=1.4,
            capsize=3,
        )
        axis.set_title(title)
        axis.set_xlabel(xlabel)
        axis.set_yticks(y_pos)
        axis.set_yticklabels(EXPECTED_METHODS)
        axis.grid(axis="x", alpha=0.25)
    fig.suptitle("Bootstrap 95% Confidence Intervals")
    fig.tight_layout()
    fig.savefig(figures_dir / "bootstrap_confidence_intervals.png")
    plt.close(fig)

    env_plot = env_method.copy()
    env_plot["env_id"] = env_plot["env_id"].astype(str)
    env_plot["method"] = env_plot["method"].astype(str)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharex=True, sharey=True)
    for axis, env_id in zip(axes, EXPECTED_ENVS, strict=False):
        subset = env_plot.loc[env_plot["env_id"] == env_id]
        for _, row in subset.iterrows():
            method = row["method"]
            axis.scatter(row["safe_rate"], row["return"], s=58, color=colors[method])
            axis.annotate(method, (row["safe_rate"], row["return"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
        axis.set_title(env_id.replace("Safety", ""))
        axis.set_xlabel("Safe rate")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Return")
    fig.suptitle("Environment-Level Return vs Safe Rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "env_tradeoff_facets.png")
    plt.close(fig)

    profile_matrix = (
        normalized_profiles.pivot(index="method", columns="metric", values="normalized_score")
        .reindex(EXPECTED_METHODS)
        .reindex([metric for metric, _, _ in PROFILE_METRICS], axis=1)
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    image = ax.imshow(profile_matrix.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(profile_matrix.columns)))
    ax.set_xticklabels(profile_matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(EXPECTED_METHODS)))
    ax.set_yticklabels(EXPECTED_METHODS)
    for i, method in enumerate(EXPECTED_METHODS):
        for j, metric in enumerate(profile_matrix.columns):
            ax.text(j, i, fmt_num(profile_matrix.loc[method, metric], 2), ha="center", va="center", color="black")
    ax.set_title("Normalized Method Profiles")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Normalized score; higher is better")
    fig.tight_layout()
    fig.savefig(figures_dir / "normalized_method_profiles.png")
    plt.close(fig)

    rank_matrix = (
        method_rank_profile.pivot(index="method", columns="metric", values="rank")
        .reindex(EXPECTED_METHODS)
        .reindex([metric for metric, _, _ in PROFILE_METRICS], axis=1)
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    image = ax.imshow(rank_matrix.to_numpy(dtype=float), cmap="viridis_r", vmin=1.0, vmax=float(len(EXPECTED_METHODS)))
    ax.set_xticks(range(len(rank_matrix.columns)))
    ax.set_xticklabels(rank_matrix.columns, rotation=30, ha="right")
    ax.set_yticks(range(len(EXPECTED_METHODS)))
    ax.set_yticklabels(EXPECTED_METHODS)
    for i, method in enumerate(EXPECTED_METHODS):
        for j, metric in enumerate(rank_matrix.columns):
            ax.text(j, i, fmt_num(rank_matrix.loc[method, metric], 1), ha="center", va="center", color="white")
    ax.set_title("Method Rank Profile")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Rank; lower is better")
    fig.tight_layout()
    fig.savefig(figures_dir / "method_metric_rank_heatmap.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    safe_values = [float(overall_plot.loc[overall_plot["method"] == method, "safe_rate_mean"].iloc[0]) for method in EXPECTED_METHODS]
    unsafe_values = [
        float(overall_plot.loc[overall_plot["method"] == method, "nonzero_cost_frequency_mean"].iloc[0])
        for method in EXPECTED_METHODS
    ]
    x = np.arange(len(EXPECTED_METHODS))
    ax.bar(x, safe_values, color="#4C78A8", label="Zero-violation episodes")
    ax.bar(x, unsafe_values, bottom=safe_values, color="#E45756", label="Nonzero-cost episodes")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(EXPECTED_METHODS, rotation=30, ha="right")
    ax.set_ylabel("Episode fraction")
    ax.set_title("Zero-Violation Gap by Method")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "zero_violation_gap_by_method.png")
    plt.close(fig)

    tradeoff_plot = relative_tradeoffs.copy()
    tradeoff_plot["method"] = tradeoff_plot["method"].astype(str)
    tradeoff_plot = tradeoff_plot.loc[tradeoff_plot["method"] != "PPO"].set_index("method").reindex(
        [method for method in EXPECTED_METHODS if method != "PPO"]
    )
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    x = np.arange(len(tradeoff_plot))
    width = 0.35
    return_retained = tradeoff_plot["return_retained_pct"].to_numpy(dtype=float)
    unsafe_reduction = tradeoff_plot["unsafe_episode_reduction_pct"].to_numpy(dtype=float)
    ax.bar(x - width / 2, return_retained, width, color="#4C78A8", label="Return retained vs PPO")
    ax.bar(x + width / 2, unsafe_reduction, width, color="#E45756", label="Unsafe episodes reduced vs PPO")
    ax.axhline(100.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.55)
    ax.set_xticks(x)
    ax.set_xticklabels(tradeoff_plot.index.tolist(), rotation=25, ha="right")
    ax.set_ylabel("Percent")
    ax.set_title("Safety Gains and Return Retention Relative to PPO")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    for idx, (retained, reduced) in enumerate(zip(return_retained, unsafe_reduction, strict=False)):
        ax.text(idx - width / 2, retained + 2.0, fmt_num(retained, 0), ha="center", va="bottom", fontsize=7)
        ax.text(idx + width / 2, reduced + 2.0, fmt_num(reduced, 0), ha="center", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(figures_dir / "relative_to_ppo_tradeoffs.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3))
    ax_left, ax_right = axes
    main_panel_offsets = {
        "PPO": (6, 5),
        "PPOLag": (6, 5),
        "FOCOPS": (6, 5),
        "CPO": (6, 5),
        "CPPOPID": (8, -12),
        "PPOSaute": (8, 8),
    }
    for _, row in overall_plot.iterrows():
        method = row["method"]
        ax_left.scatter(row["safe_rate_mean"], row["return_mean"], s=72, color=colors[method], edgecolor="black", linewidth=0.5)
        ax_left.annotate(
            method,
            (row["safe_rate_mean"], row["return_mean"]),
            xytext=main_panel_offsets.get(method, (5, 4)),
            textcoords="offset points",
            fontsize=8,
        )
    ax_left.set_xlabel("Safe rate")
    ax_left.set_ylabel("Return")
    ax_left.set_title("Return vs zero-violation frequency")
    ax_left.grid(True, alpha=0.25)

    ax_right.bar(x - width / 2, return_retained, width, color="#4C78A8", label="Return retained")
    ax_right.bar(x + width / 2, unsafe_reduction, width, color="#E45756", label="Unsafe episodes reduced")
    ax_right.axhline(100.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.45)
    ax_right.set_xticks(x)
    ax_right.set_xticklabels(tradeoff_plot.index.tolist(), rotation=25, ha="right")
    ax_right.set_ylim(0.0, 105.0)
    ax_right.set_ylabel("Percent vs PPO")
    ax_right.set_title("Cost of unsafe-episode reduction")
    ax_right.legend(loc="upper left", fontsize=8)
    ax_right.grid(axis="y", alpha=0.25)
    for idx, (retained, reduced) in enumerate(zip(return_retained, unsafe_reduction, strict=False)):
        ax_right.text(idx - width / 2, retained + 1.8, fmt_num(retained, 0), ha="center", va="bottom", fontsize=7)
        ax_right.text(idx + width / 2, reduced + 1.8, fmt_num(reduced, 0), ha="center", va="bottom", fontsize=7)
    fig.suptitle("Reward-Safety Trade-off Under Episode-Level Zero-Violation Metrics", fontsize=12)
    fig.subplots_adjust(left=0.07, right=0.985, bottom=0.20, top=0.82, wspace=0.24)
    fig.savefig(figures_dir / "tradeoff_main_panel.png")
    plt.close(fig)

    boundary_plot = claim_boundary.copy()
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Claim Boundary", fontsize=13, weight="bold", pad=10)
    columns = [
        ("Paper-safe claim", "claim", 0.03, 0.25, 34),
        ("Evidence", "evidence", 0.29, 0.27, 36),
        ("Boundary", "boundary", 0.58, 0.19, 27),
        ("Not supported", "not_supported", 0.79, 0.18, 25),
    ]
    header_y = 0.90
    row_height = 0.132
    for label, _, x0, width, _ in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x0, header_y),
                width,
                0.055,
                boxstyle="round,pad=0.006,rounding_size=0.01",
                linewidth=0.8,
                edgecolor="#333333",
                facecolor="#E8EEF7",
            )
        )
        ax.text(x0 + 0.008, header_y + 0.028, label, va="center", ha="left", fontsize=9, weight="bold")
    for row_idx, (_, row) in enumerate(boundary_plot.iterrows()):
        y0 = header_y - (row_idx + 1) * row_height
        fill = "#FFFFFF" if row_idx % 2 == 0 else "#F7F7F7"
        for _, column, x0, width, wrap_width in columns:
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y0),
                    width,
                    row_height - 0.012,
                    boxstyle="round,pad=0.006,rounding_size=0.008",
                    linewidth=0.45,
                    edgecolor="#CCCCCC",
                    facecolor=fill,
                )
            )
            color = "#8C2D2D" if column == "not_supported" else "#222222"
            ax.text(
                x0 + 0.008,
                y0 + row_height - 0.034,
                wrap_text(row[column], wrap_width),
                va="top",
                ha="left",
                fontsize=7.1,
                color=color,
                linespacing=1.12,
            )
    fig.tight_layout()
    fig.savefig(figures_dir / "claim_boundary.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Four Main Findings for the Paper Narrative", fontsize=13, weight="bold", pad=10)
    finding_colors = ["#4C78A8", "#E45756", "#54A24B", "#6F6F6F"]
    box_height = 0.185
    y_start = 0.76
    for idx, (_, row) in enumerate(main_findings_summary.iterrows()):
        y0 = y_start - idx * 0.205
        color = finding_colors[idx % len(finding_colors)]
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y0),
                0.12,
                box_height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=color,
                facecolor=color,
            )
        )
        ax.text(
            0.095,
            y0 + box_height / 2,
            str(row["finding_id"]),
            ha="center",
            va="center",
            fontsize=18,
            weight="bold",
            color="#FFFFFF",
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.18, y0),
                0.78,
                box_height,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.0,
                edgecolor="#CCCCCC",
                facecolor="#FFFFFF",
            )
        )
        ax.text(
            0.205,
            y0 + box_height - 0.036,
            wrap_text(row["finding"], 82),
            ha="left",
            va="top",
            fontsize=9.2,
            weight="bold",
            color=color,
        )
        ax.text(
            0.205,
            y0 + box_height - 0.082,
            wrap_text(row["evidence"], 106),
            ha="left",
            va="top",
            fontsize=7.8,
            color="#222222",
        )
        ax.text(
            0.205,
            y0 + 0.032,
            wrap_text(row["paper_use"], 106),
            ha="left",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
    fig.tight_layout()
    fig.savefig(figures_dir / "main_findings_summary.png")
    plt.close(fig)

    metric_rows = metric_family_map.copy()
    family_colors = ["#4C78A8", "#54A24B", "#E45756", "#F58518", "#B279A2"]
    fig, ax = plt.subplots(figsize=(12.4, 6.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Metric Families and the Questions They Answer", fontsize=13, weight="bold", pad=10)
    row_height = 0.16
    y_start = 0.78
    for idx, (_, row) in enumerate(metric_rows.iterrows()):
        y0 = y_start - idx * 0.17
        color = family_colors[idx % len(family_colors)]
        ax.add_patch(
            FancyBboxPatch(
                (0.035, y0),
                0.18,
                row_height,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=color,
                facecolor="#FFFFFF",
            )
        )
        ax.text(
            0.125,
            y0 + row_height / 2,
            wrap_text(row["family"], 20),
            ha="center",
            va="center",
            fontsize=9,
            weight="bold",
            color=color,
        )
        ax.add_patch(
            FancyBboxPatch(
                (0.24, y0),
                0.27,
                row_height,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                linewidth=0.7,
                edgecolor="#CCCCCC",
                facecolor="#F7F7F7",
            )
        )
        ax.text(0.255, y0 + row_height - 0.035, "Metrics", ha="left", va="top", fontsize=7.6, weight="bold")
        ax.text(0.255, y0 + row_height - 0.065, wrap_text(row["metrics"], 38), ha="left", va="top", fontsize=7.4)
        ax.add_patch(
            FancyBboxPatch(
                (0.535, y0),
                0.42,
                row_height,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                linewidth=0.7,
                edgecolor="#CCCCCC",
                facecolor="#FFFFFF",
            )
        )
        ax.text(0.55, y0 + row_height - 0.035, "Evaluation question", ha="left", va="top", fontsize=7.6, weight="bold")
        ax.text(0.55, y0 + row_height - 0.065, wrap_text(row["evaluation_question"], 60), ha="left", va="top", fontsize=7.4)
        ax.text(0.55, y0 + 0.032, wrap_text(row["paper_role"], 64), ha="left", va="bottom", fontsize=7.1, color="#555555")
    fig.tight_layout()
    fig.savefig(figures_dir / "metric_family_map.png")
    plt.close(fig)

    quadrant_plot = method_tradeoff_quadrants.copy()
    quadrant_plot["method"] = quadrant_plot["method"].astype(str)
    fig, ax = plt.subplots(figsize=(7.4, 5.4))
    return_threshold = float(quadrant_plot["return_threshold"].iloc[0])
    safe_threshold = float(quadrant_plot["safe_rate_threshold"].iloc[0])
    ax.axvline(safe_threshold, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    ax.axhline(return_threshold, color="#777777", linestyle="--", linewidth=1.0, alpha=0.75)
    label_offsets = {
        "PPO": (6, 5),
        "PPOLag": (6, -2),
        "FOCOPS": (6, 5),
        "CPO": (6, 5),
        "CPPOPID": (8, -10),
        "PPOSaute": (8, 8),
    }
    for _, row in quadrant_plot.iterrows():
        method = row["method"]
        ax.scatter(row["safe_rate_mean"], row["return_mean"], s=90, color=colors[method], edgecolor="black", linewidth=0.6)
        ax.annotate(
            method,
            (row["safe_rate_mean"], row["return_mean"]),
            xytext=label_offsets.get(method, (6, 5)),
            textcoords="offset points",
            fontsize=8,
        )
    ax.text(0.03, 0.94, "higher return\nlower safe rate", transform=ax.transAxes, va="top", ha="left", fontsize=8, color="#555555")
    ax.text(0.97, 0.94, "higher return\nhigher safe rate", transform=ax.transAxes, va="top", ha="right", fontsize=8, color="#555555")
    ax.text(0.03, 0.06, "lower return\nlower safe rate", transform=ax.transAxes, va="bottom", ha="left", fontsize=8, color="#555555")
    ax.text(0.97, 0.06, "lower return\nhigher safe rate", transform=ax.transAxes, va="bottom", ha="right", fontsize=8, color="#555555")
    ax.set_xlabel("Safe rate")
    ax.set_ylabel("Return")
    ax.set_title("Method Trade-Off Quadrants")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures_dir / "method_tradeoff_quadrants.png")
    plt.close(fig)

    env_profile = env_method.copy()
    env_profile["env_id"] = env_profile["env_id"].astype(str)
    env_profile["method"] = env_profile["method"].astype(str)
    metrics = ["return", "safe_rate", "p95_cost", "max_consecutive_cost_run"]
    labels = ["return", "safe rate", "p95 cost", "max run"]
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 6.8))
    axes_flat = axes.ravel()
    x = np.arange(len(EXPECTED_ENVS))
    width = 0.12
    offsets = (np.arange(len(EXPECTED_METHODS)) - (len(EXPECTED_METHODS) - 1) / 2) * width
    for axis, metric, label in zip(axes_flat, metrics, labels, strict=False):
        for idx, method in enumerate(EXPECTED_METHODS):
            values = []
            for env_id in EXPECTED_ENVS:
                value = env_profile.loc[
                    (env_profile["env_id"] == env_id) & (env_profile["method"] == method),
                    metric,
                ].iloc[0]
                values.append(float(value))
            axis.bar(x + offsets[idx], values, width=width, color=colors[method], label=method if metric == "return" else None)
        axis.set_title(label)
        axis.set_xticks(x)
        axis.set_xticklabels([env.replace("Safety", "").replace("-v0", "") for env in EXPECTED_ENVS], rotation=18, ha="right")
        axis.grid(axis="y", alpha=0.25)
    axes_flat[0].legend(ncol=3, fontsize=7, loc="upper left")
    fig.suptitle("Environment-Specific Metric Profiles")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(figures_dir / "environment_metric_profiles.png")
    plt.close(fig)

    bubble_plot = overall_plot.copy()
    fig, ax = plt.subplots(figsize=(7.6, 5.1))
    sizes = 90 + 520 * bubble_plot["nonzero_cost_frequency_mean"].astype(float)
    for size, (_, row) in zip(sizes, bubble_plot.iterrows(), strict=False):
        method = row["method"]
        ax.scatter(
            row["safe_rate_mean"],
            row["return_mean"],
            s=float(size),
            color=colors[method],
            alpha=0.72,
            edgecolor="black",
            linewidth=0.7,
        )
        ax.annotate(method, (row["safe_rate_mean"], row["return_mean"]), xytext=(6, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Safe rate")
    ax.set_ylabel("Return")
    ax.set_title("Three-Axis Trade-Off: Return, Safe Rate, and Unsafe-Episode Frequency")
    ax.grid(True, alpha=0.25)
    ax.text(
        0.03,
        0.04,
        "bubble size = nonzero-cost episode frequency",
        transform=ax.transAxes,
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    fig.savefig(figures_dir / "three_axis_tradeoff_bubble.png")
    plt.close(fig)

    zero_gap = env_method.copy()
    zero_gap["env_id"] = zero_gap["env_id"].astype(str)
    zero_gap["method"] = zero_gap["method"].astype(str)
    zero_gap["zero_violation_gap"] = 1.0 - zero_gap["safe_rate"].astype(float)
    gap_heat = zero_gap.pivot(index="env_id", columns="method", values="zero_violation_gap").reindex(EXPECTED_ENVS)
    gap_heat = gap_heat[list(EXPECTED_METHODS)]
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    image = ax.imshow(gap_heat.to_numpy(dtype=float), cmap="Reds", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(EXPECTED_METHODS)))
    ax.set_xticklabels(EXPECTED_METHODS, rotation=30, ha="right")
    ax.set_yticks(range(len(EXPECTED_ENVS)))
    ax.set_yticklabels(EXPECTED_ENVS)
    for i, env_id in enumerate(EXPECTED_ENVS):
        for j, method in enumerate(EXPECTED_METHODS):
            ax.text(j, i, fmt_num(gap_heat.loc[env_id, method], 2), ha="center", va="center", color="black")
    ax.set_title("Environment-Method Zero-Violation Gap")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("1 - safe rate")
    fig.tight_layout()
    fig.savefig(figures_dir / "env_zero_violation_gap_heatmap.png")
    plt.close(fig)

    signature_plot = (
        method_safety_signature.pivot(index="method", columns="safety_dimension", values="normalized_score")
        .reindex(EXPECTED_METHODS)
        .reindex(["safe_rate", "low_nonzero_frequency", "low_mean_cost", "low_p95_cost", "short_max_run"], axis=1)
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    image = ax.imshow(signature_plot.to_numpy(dtype=float), cmap="YlGnBu", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(signature_plot.columns)))
    ax.set_xticklabels(signature_plot.columns, rotation=28, ha="right")
    ax.set_yticks(range(len(EXPECTED_METHODS)))
    ax.set_yticklabels(EXPECTED_METHODS)
    for i, method in enumerate(EXPECTED_METHODS):
        for j, dimension in enumerate(signature_plot.columns):
            ax.text(j, i, fmt_num(signature_plot.loc[method, dimension], 2), ha="center", va="center", color="black")
    ax.set_title("Method Safety Signatures")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Normalized score; higher is safer")
    fig.tight_layout()
    fig.savefig(figures_dir / "method_safety_signature.png")
    plt.close(fig)

    flow_plot = claim_flow_rows.copy()
    fig, ax = plt.subplots(figsize=(12.0, 3.8))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Claim-Evidence Flow for the Empirical Study", fontsize=13, weight="bold", pad=8)
    xs = np.linspace(0.08, 0.92, len(flow_plot))
    for idx, (_, row) in enumerate(flow_plot.iterrows()):
        x0 = float(xs[idx])
        y0 = 0.50
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 0.075, y0 - 0.17),
                0.15,
                0.34,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.0,
                edgecolor="#4C78A8",
                facecolor="#FFFFFF",
            )
        )
        ax.text(x0, y0 + 0.105, f"Stage {row['stage']}", ha="center", va="center", fontsize=9, weight="bold", color="#4C78A8")
        ax.text(x0, y0 + 0.035, wrap_text(row["component"], 18), ha="center", va="center", fontsize=8.2, weight="bold")
        ax.text(x0, y0 - 0.075, wrap_text(row["paper_role"], 22), ha="center", va="center", fontsize=7.2, color="#555555")
        if idx < len(flow_plot) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + 0.082, y0),
                    (float(xs[idx + 1]) - 0.082, y0),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.0,
                    color="#777777",
                )
            )
    fig.tight_layout()
    fig.savefig(figures_dir / "claim_evidence_flow.png")
    plt.close(fig)

    upgrade_plot = reporting_protocol_upgrade.copy()
    fig, ax = plt.subplots(figsize=(12.4, 5.2))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("From Return-Cost Reporting to Episode-Level Zero-Violation Reporting", fontsize=13, weight="bold", pad=8)
    ax.text(
        0.50,
        0.90,
        "The contribution is a reporting protocol: each added layer answers a safety question left open by the previous layer.",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#333333",
    )
    xs = np.linspace(0.10, 0.90, len(upgrade_plot))
    palette = ["#4C78A8", "#54A24B", "#F58518", "#B279A2", "#E45756"]
    for idx, (_, row) in enumerate(upgrade_plot.iterrows()):
        x0 = float(xs[idx])
        color = palette[idx % len(palette)]
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 0.082, 0.48),
                0.164,
                0.265,
                boxstyle="round,pad=0.012,rounding_size=0.018",
                linewidth=1.1,
                edgecolor=color,
                facecolor="#FFFFFF",
            )
        )
        ax.text(x0, 0.704, f"Layer {idx + 1}", ha="center", va="center", fontsize=8, weight="bold", color=color)
        ax.text(
            x0,
            0.640,
            wrap_text(row["reporting_layer"], 20),
            ha="center",
            va="center",
            fontsize=8.1,
            weight="bold",
            color="#222222",
        )
        ax.text(
            x0,
            0.545,
            wrap_text(row["reported_quantities"], 24),
            ha="center",
            va="center",
            fontsize=7.0,
            color="#444444",
        )
        ax.add_patch(
            FancyBboxPatch(
                (x0 - 0.082, 0.18),
                0.164,
                0.20,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                linewidth=0.8,
                edgecolor="#CCCCCC",
                facecolor="#F8F8F8",
            )
        )
        ax.text(
            x0,
            0.305,
            wrap_text(row["safety_question_answered"], 26),
            ha="center",
            va="center",
            fontsize=6.9,
            color="#222222",
        )
        ax.text(
            x0,
            0.220,
            wrap_text(row["paper_use"], 28),
            ha="center",
            va="center",
            fontsize=6.5,
            color="#555555",
        )
        if idx < len(upgrade_plot) - 1:
            ax.add_patch(
                FancyArrowPatch(
                    (x0 + 0.09, 0.61),
                    (float(xs[idx + 1]) - 0.09, 0.61),
                    arrowstyle="-|>",
                    mutation_scale=12,
                    linewidth=1.0,
                    color="#777777",
                )
            )
    fig.tight_layout()
    fig.savefig(figures_dir / "reporting_protocol_upgrade.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    case_plot = environment_case_studies.copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.8), sharey=True)
    case_colors = ["#4C78A8", "#F58518", "#54A24B"]
    for idx, (axis, (_, row)) in enumerate(zip(axes, case_plot.iterrows(), strict=False)):
        axis.set_xlim(0.0, 1.0)
        axis.set_ylim(0.0, 1.0)
        axis.axis("off")
        color = case_colors[idx % len(case_colors)]
        axis.add_patch(
            FancyBboxPatch(
                (0.04, 0.06),
                0.92,
                0.88,
                boxstyle="round,pad=0.014,rounding_size=0.018",
                linewidth=1.2,
                edgecolor=color,
                facecolor="#FFFFFF",
            )
        )
        axis.text(
            0.50,
            0.885,
            row["env_id"].replace("Safety", "").replace("-v0", ""),
            ha="center",
            va="center",
            fontsize=9.2,
            weight="bold",
            color=color,
        )
        items = [
            ("Return leader", row["return_leader"]),
            ("Safe-rate leader", row["safe_rate_leader"]),
            ("Mean-cost leader", row["mean_cost_leader"]),
            ("Tail leader", row["tail_leader"]),
            ("Best ZV gap", fmt_num(row["zero_violation_gap_best"])),
            ("FOCOPS", row["focops_profile"]),
        ]
        y0 = 0.765
        for label, value in items:
            axis.text(0.09, y0, label, ha="left", va="top", fontsize=7.4, weight="bold", color="#333333")
            axis.text(0.45, y0, wrap_text(value, 25), ha="left", va="top", fontsize=7.2, color="#222222")
            y0 -= 0.095 if label != "FOCOPS" else 0.145
        axis.add_patch(
            FancyBboxPatch(
                (0.09, 0.105),
                0.82,
                0.18,
                boxstyle="round,pad=0.010,rounding_size=0.014",
                linewidth=0.7,
                edgecolor="#DDDDDD",
                facecolor="#F7F7F7",
            )
        )
        axis.text(
            0.12,
            0.195,
            wrap_text(row["interpretation"], 43),
            ha="left",
            va="center",
            fontsize=7.0,
            color="#333333",
        )
    fig.suptitle("Environment Case Studies for Main Interpretation", fontsize=13, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(figures_dir / "environment_case_studies.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    coverage_plot = protocol_coverage_matrix.copy()
    fig, ax = plt.subplots(figsize=(12.8, 6.4))
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.axis("off")
    ax.set_title("Protocol Coverage and Claim Boundary", fontsize=13, weight="bold", pad=8)
    ax.text(
        0.50,
        0.925,
        "Each coverage axis supports a scoped claim and blocks a stronger unsupported reading.",
        ha="center",
        va="center",
        fontsize=8.5,
        color="#333333",
    )
    headers = [
        ("Coverage axis", 0.04, 0.16),
        ("Reported coverage", 0.22, 0.20),
        ("Claim supported", 0.45, 0.23),
        ("Claim boundary", 0.72, 0.23),
    ]
    header_y = 0.84
    for label, x0, width in headers:
        ax.add_patch(
            FancyBboxPatch(
                (x0, header_y),
                width,
                0.055,
                boxstyle="round,pad=0.006,rounding_size=0.010",
                linewidth=0.8,
                edgecolor="#333333",
                facecolor="#E8EEF7",
            )
        )
        ax.text(x0 + 0.008, header_y + 0.028, label, ha="left", va="center", fontsize=8.6, weight="bold")
    row_height = 0.118
    for row_idx, (_, row) in enumerate(coverage_plot.iterrows()):
        y0 = header_y - (row_idx + 1) * row_height
        fill = "#FFFFFF" if row_idx % 2 == 0 else "#FAFAFA"
        cells = [
            (row["coverage_axis"], 0.04, 0.16, 20, "#222222", True),
            (row["reported_coverage"], 0.22, 0.20, 31, "#222222", False),
            (row["claim_supported"], 0.45, 0.23, 35, "#1F5A2D", False),
            (row["claim_boundary"], 0.72, 0.23, 35, "#8C2D2D", False),
        ]
        for text, x0, width, wrap_width, color, bold in cells:
            ax.add_patch(
                FancyBboxPatch(
                    (x0, y0),
                    width,
                    row_height - 0.012,
                    boxstyle="round,pad=0.006,rounding_size=0.008",
                    linewidth=0.45,
                    edgecolor="#CCCCCC",
                    facecolor=fill,
                )
            )
            ax.text(
                x0 + 0.008,
                y0 + row_height - 0.032,
                wrap_text(text, wrap_width),
                ha="left",
                va="top",
                fontsize=7.1,
                color=color,
                weight="bold" if bold else "normal",
                linespacing=1.10,
            )
    fig.tight_layout()
    fig.savefig(figures_dir / "protocol_coverage_matrix.png", dpi=220, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def render_evidence_note(
    validation: dict[str, Any],
    overall: pd.DataFrame,
    rankings: pd.DataFrame,
    safety_rankings: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    relative_tradeoffs: pd.DataFrame,
    literature_metric_coverage: pd.DataFrame,
) -> str:
    ppo = row_for(overall, "PPO")
    focops = row_for(overall, "FOCOPS")
    cppopid = row_for(overall, "CPPOPID")
    pporsaute = row_for(overall, "PPOSaute")
    best_safe_rate = max(float(cppopid["safe_rate_mean"]), float(pporsaute["safe_rate_mean"]))
    best_nonzero = 1.0 - best_safe_rate
    ranking_text = markdown_table(
        rankings.rename(columns={"mean_rank": "rank"}).assign(method=lambda frame: frame["method"].astype(str)),
        [("method", "method"), ("mean rank", "rank")],
    )
    safety_ranking_text = markdown_table(
        safety_rankings.rename(columns={"mean_rank": "rank"}).assign(method=lambda frame: frame["method"].astype(str)),
        [("method", "method"), ("mean rank", "rank")],
    )
    return_safe_corr = float(metric_correlation.loc["return", "safe_rate"])
    mean_cost_safe_corr = float(metric_correlation.loc["mean_cost", "safe_rate"])
    focops_safe_ci = ci_for(bootstrap_ci, "FOCOPS", "safe_rate")
    ppo_return_ci = ci_for(bootstrap_ci, "PPO", "return")
    focops_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "FOCOPS"].iloc[0]
    cppopid_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "CPPOPID"].iloc[0]
    pporsaute_tradeoff = relative_tradeoffs.loc[relative_tradeoffs["method"] == "PPOSaute"].iloc[0]
    literature_coverage_text = markdown_table(
        literature_metric_coverage,
        [
            ("research line", "research_line"),
            ("expected cost", "expected_cost"),
            ("episode event", "episode_event"),
            ("tail severity", "tail_severity"),
            ("persistence", "temporal_persistence"),
            ("protocol reliability", "protocol_reliability"),
        ],
        digits=2,
    )
    return f"""# Paper Evidence Pack

## Purpose

The aggregate evaluation matrix provides the main evidence package for the Safe RL zero-violation metrics study. The evidence evaluates mature Safe RL baselines rather than a new-method positive result.

## Data Source

| field | value |
| --- | ---: |
| rows | {validation["rows"]} |
| methods | {len(validation["methods"])} |
| environments | {len(validation["envs"])} |
| seeds | {len(validation["seeds"])} |
| successful runs | {validation["completed"]} |
| successful evaluations | {validation["completed_evaluations"]} |
| training steps per run | {EXPECTED_STEPS} |

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

PPO has the highest overall return (`{fmt_num(ppo["return_mean"])}`) but the weakest safety profile: safe rate `{fmt_num(ppo["safe_rate_mean"])}` and nonzero-cost frequency `{fmt_num(ppo["nonzero_cost_frequency_mean"])}`.

CPPOPID and PPOSaute are the strongest zero-violation-rate baselines. Their safe rates are `{fmt_num(cppopid["safe_rate_mean"])}` and `{fmt_num(pporsaute["safe_rate_mean"])}`. Even the stronger of these still leaves an estimated `{fmt_num(best_nonzero)}` nonzero-cost episode frequency.

FOCOPS remains the main balanced baseline. It has return `{fmt_num(focops["return_mean"])}`, mean cost `{fmt_num(focops["mean_cost_mean"])}`, safe rate `{fmt_num(focops["safe_rate_mean"])}`, and conditional unsafe severity `{fmt_num(focops["conditional_unsafe_severity_mean"])}`.

Relative-to-PPO diagnostics make the reward-safety exchange explicit. FOCOPS retains `{fmt_num(focops_tradeoff["return_retained_pct"])}`% of PPO return while reducing unsafe-episode frequency by `{fmt_num(focops_tradeoff["unsafe_episode_reduction_pct"])}`%. CPPOPID retains `{fmt_num(cppopid_tradeoff["return_retained_pct"])}`% of PPO return and reduces unsafe episodes by `{fmt_num(cppopid_tradeoff["unsafe_episode_reduction_pct"])}`%. PPOSaute retains `{fmt_num(pporsaute_tradeoff["return_retained_pct"])}`% of PPO return and reduces unsafe episodes by `{fmt_num(pporsaute_tradeoff["unsafe_episode_reduction_pct"])}`%.

The run-level return / safe-rate Pearson correlation is `{fmt_num(return_safe_corr)}`. The mean-cost / safe-rate Pearson correlation is `{fmt_num(mean_cost_safe_corr)}`. These diagnostic correlations support the paper framing that safety metrics should not be collapsed into a single expected-cost number.

Bootstrap intervals provide uncertainty context without adding new training. PPO return mean is `{fmt_num(ppo_return_ci["mean"])}` with 95% CI `[{fmt_num(ppo_return_ci["ci95_low"])}, {fmt_num(ppo_return_ci["ci95_high"])}]`. FOCOPS safe-rate mean is `{fmt_num(focops_safe_ci["mean"])}` with 95% CI `[{fmt_num(focops_safe_ci["ci95_low"])}, {fmt_num(focops_safe_ci["ci95_high"])}]`.

## Literature-to-Metric Coverage

The coverage map is a qualitative writing aid for related-work positioning. It shows that adjacent papers cover optimizers, benchmarks, implementation infrastructure, risk objectives, zero-violation methods, and statistical reliability, while the empirical study places all reporting dimensions in one baseline matrix.

{literature_coverage_text}

## Ranking Summary

Mean rank across return and safety metrics:

{ranking_text}
Safety-focused mean rank:

{safety_ranking_text}
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
"""


def row_for(overall: pd.DataFrame, method: str) -> pd.Series:
    return overall.loc[overall["method"].astype(str) == method].iloc[0]


def ci_for(bootstrap_ci: pd.DataFrame, method: str, metric: str) -> pd.Series:
    return bootstrap_ci.loc[(bootstrap_ci["method"] == method) & (bootstrap_ci["metric"] == metric)].iloc[0]


def render_outline() -> str:
    return """# Zero-Violation Metrics Empirical Study Skeleton

## Working Title

Beyond Expected Cost: An Empirical Study of Episode-Level Zero-Violation Metrics in Safe Reinforcement Learning

## Abstract Claim

Expected cumulative cost is not sufficient to characterize episode-level safety. Mature Safe RL baselines reduce safety costs in different ways, but mean cost, zero-violation probability, tail severity, and violation run length can diverge.

## Introduction

- Safe RL typically reports return and expected cumulative cost.
- Deployment safety often depends on whether an episode contains any violation.
- Episode-level zero-violation metrics expose a different safety axis.
- The study evaluates mature baselines rather than proposing a primary new algorithm.

## Metrics

- Return.
- Mean cost and violation rate.
- Safe rate and nonzero-cost episode frequency.
- p90, p95, and max episode cost.
- Conditional unsafe severity.
- Max consecutive cost run.

## Experimental Setup

- Algorithms: PPO, PPOLag, FOCOPS, CPO, CPPOPID, PPOSaute.
- Environments: SafetyPointGoal1-v0, SafetyPointButton1-v0, SafetyCarGoal1-v0.
- Seeds: 1, 2, 3.
- Training budget: 5,000,000 steps per run.
- Evaluation: 50 episodes, 200 max steps.

## Results

- PPO provides the reward baseline and illustrates unsafe high-return behavior.
- CPPOPID and PPOSaute provide the strongest safe-rate baselines with weak return.
- FOCOPS provides the strongest balanced baseline.
- No mature baseline reaches actual zero-violation behavior.
- Mean cost, safe rate, tail severity, and run length show non-identical ranking behavior.
- Environment-level trade-off facets and method-profile heatmaps summarize where the ranking changes.
- Metric-family, quadrant, and environment-profile figures turn the aggregate matrix into paper evidence rather than an execution trace.
- Three-axis bubble, zero-violation-gap heatmap, safety-signature, and claim-flow figures strengthen the paper-facing evidence chain.
- Statistical-reporting checklist and ladder artifacts connect the evidence matrix to RL evaluation guidance on uncertainty, seed sensitivity, and scoped statistical interpretation.
- Reporting-protocol upgrade, environment-case-study, environment-method scorecard, and protocol-coverage artifacts convert the matrix into paper evidence.

## Discussion

- Expected-cost constraints and episode-level event constraints target different objects.
- A low mean cost can coexist with nonzero episode violation probability.
- A high safe rate can coexist with severe tail failures.
- Zero-violation metrics should be reported alongside expected cost in Safe RL evaluations.
- Literature positioning separates expected-cost optimization, intervention mechanisms, risk objectives, implementation infrastructure, verified safety, and benchmark protocols from the paper's evaluation-layer contribution.
- The literature-to-metric coverage figure makes the reporting gap explicit by mapping adjacent paper families to expected-cost, event-frequency, tail, persistence, and protocol-reliability dimensions.
- Protocol-upgrade artifacts explain how the paper extends return-cost reporting into episode-event, tail, persistence, and claim-boundary reporting layers.
- Statistical comparison should remain descriptive and scope-aware because the matrix has three seeds per environment-method cell rather than enough independent runs for broad dominance claims.
- Reproducibility material should be presented as supporting material for aggregate evidence inspection, not as the paper's primary narrative.

## Limitations

- The study covers six algorithms, three environments, and three seeds.
- The result does not cover all Safe RL methods.
- The result does not establish a new method.
- WCSAC and external tail-risk implementations require separate integration if included.

## Reproducibility

- The study uses a reported 54-cell evaluation matrix.
- Derived tables and figures are generated from aggregate evaluation metrics.
- Reproducibility material is sufficient to reproduce the reported aggregate tables and figures.
- Prototype methods and exploratory variants remain outside the main result boundary.
"""


def write_notes(
    repo_root: Path,
    validation: dict[str, Any],
    overall: pd.DataFrame,
    rankings: pd.DataFrame,
    safety_rankings: pd.DataFrame,
    metric_correlation: pd.DataFrame,
    bootstrap_ci: pd.DataFrame,
    relative_tradeoffs: pd.DataFrame,
    literature_positioning_map: pd.DataFrame,
    literature_metric_coverage: pd.DataFrame,
) -> None:
    notes_dir = repo_root / "notes"
    outline_dir = repo_root / "outline"
    notes_dir.mkdir(parents=True, exist_ok=True)
    outline_dir.mkdir(parents=True, exist_ok=True)
    (notes_dir / "evidence_summary.md").write_text(
        render_evidence_note(
            validation,
            overall,
            rankings,
            safety_rankings,
            metric_correlation,
            bootstrap_ci,
            relative_tradeoffs,
            literature_metric_coverage,
        ),
        encoding="utf-8",
    )
    (outline_dir / "zero_violation_empirical_study_skeleton.md").write_text(render_outline(), encoding="utf-8")


def build_artifacts(rows: list[dict[str, Any]], repo_root: Path, validation: dict[str, Any]) -> None:
    df = prepare_dataframe(rows)
    overall = aggregate_overall(df)
    env_method = aggregate_env_method(df)
    seed_variability = aggregate_seed_variability(df)
    metric_correlation = compute_metric_correlation(df)
    bootstrap_ci = compute_bootstrap_ci(df)
    environment_best = build_environment_best(env_method)
    method_rank_profile = build_method_rank_profile(overall)
    normalized_profiles = build_normalized_profiles(overall)
    relative_tradeoffs = build_relative_to_ppo_tradeoffs(overall)
    claim_boundary = build_claim_boundary(overall, metric_correlation, relative_tradeoffs)
    main_findings_summary = build_main_findings_summary(overall, metric_correlation, relative_tradeoffs)
    literature_positioning_map = build_literature_positioning_map()
    paper_positioning_matrix = build_paper_positioning_matrix()
    literature_metric_coverage = build_literature_metric_coverage()
    metric_family_map = build_metric_family_map()
    method_tradeoff_quadrants = build_method_tradeoff_quadrants(overall)
    method_safety_signature = build_method_safety_signature(overall)
    claim_flow_rows = build_claim_flow_rows()
    key_numbers = build_key_numbers(overall, metric_correlation, relative_tradeoffs)
    metric_disagreement = build_metric_disagreement(overall)
    statistical_reporting_checklist = build_statistical_reporting_checklist()
    reporting_protocol_upgrade = build_reporting_protocol_upgrade()
    environment_case_studies = build_environment_case_studies(env_method)
    env_method_scorecard = build_env_method_scorecard(env_method)
    protocol_coverage_matrix = build_protocol_coverage_matrix(validation)
    rankings, safety_rankings = build_rankings(env_method)
    write_tables(
        repo_root,
        overall,
        env_method,
        rankings,
        safety_rankings,
        seed_variability,
        metric_correlation,
        bootstrap_ci,
        environment_best,
        method_rank_profile,
        relative_tradeoffs,
        claim_boundary,
        main_findings_summary,
        literature_positioning_map,
        paper_positioning_matrix,
        literature_metric_coverage,
        metric_family_map,
        method_tradeoff_quadrants,
        method_safety_signature,
        claim_flow_rows,
        key_numbers,
        metric_disagreement,
        statistical_reporting_checklist,
        reporting_protocol_upgrade,
        environment_case_studies,
        env_method_scorecard,
        protocol_coverage_matrix,
    )
    write_figures(
        repo_root,
        env_method,
        overall,
        seed_variability,
        metric_correlation,
        bootstrap_ci,
        method_rank_profile,
        normalized_profiles,
        relative_tradeoffs,
        claim_boundary,
        main_findings_summary,
        literature_positioning_map,
        paper_positioning_matrix,
        literature_metric_coverage,
        metric_family_map,
        method_tradeoff_quadrants,
        method_safety_signature,
        claim_flow_rows,
        key_numbers,
        metric_disagreement,
        statistical_reporting_checklist,
        reporting_protocol_upgrade,
        environment_case_studies,
        env_method_scorecard,
        protocol_coverage_matrix,
    )
    write_notes(
        repo_root,
        validation,
        overall,
        rankings,
        safety_rankings,
        metric_correlation,
        bootstrap_ci,
        relative_tradeoffs,
        literature_positioning_map,
        literature_metric_coverage,
    )


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    rows = read_metric_rows(args.input_path)
    validation = validate_rows(rows)
    print("metric_archive_valid=true")
    print(f"rows={validation['rows']}")
    print(f"methods={len(validation['methods'])}")
    print(f"envs={len(validation['envs'])}")
    print(f"seeds={len(validation['seeds'])}")
    print(f"completed={validation['completed']}")
    print(f"completed_evaluations={validation['completed_evaluations']}")
    if args.check_only:
        print("check_only=true")
        return 0
    build_artifacts(rows, args.repo_root, validation)
    print("evidence_artifacts_saved=true")
    print(f"artifact_root={args.repo_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
