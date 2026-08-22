from __future__ import annotations

from argparse import ArgumentParser, Namespace
from copy import deepcopy
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import re
import shutil
import zipfile


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_SOURCE_SHA256 = "655efab35713898d0f281d438ab21b09eb6253eb084c4b7ac10854d416a7ea56"
EXPECTED_METHODS = ("PPO", "PPOLag", "FOCOPS", "CPO", "CPPOPID", "PPOSaute")
EXPECTED_ENVS = ("SafetyPointGoal1-v0", "SafetyPointButton1-v0", "SafetyCarGoal1-v0")
EXPECTED_SEEDS = (1, 2, 3)
EXPECTED_RUNS = 54
EXPECTED_EPISODES_PER_RUN = 50
EXPECTED_EPISODES = EXPECTED_RUNS * EXPECTED_EPISODES_PER_RUN
REQUIRED_RUN_FILES = ("config.json", "run_spec.json", "progress.csv", "evaluation.json")
DROP_HOST_KEYS = {"python_executable", "repo_root", "batch_root"}
RUN_PATH_KEYS = {
    "run_dir",
    "log_dir",
    "output_json",
    "output_md",
    "stdout_log",
    "stderr_log",
    "status_json",
}
TEXT_EXTENSIONS = {".json", ".log", ".csv", ".md", ".marker"}
WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9])[A-Z]:[\\/][^\s\"']+")
PRIVATE_TEXT_RE = re.compile(
    r"(?i)((?<![A-Za-z0-9])[A-Z]:[\\/]|/home/|/Users/|@ahut\.edu\.cn|api[_-]?key|access[_-]?token|password|passwd)"
)
FIXED_ZIP_TIME = (2026, 8, 22, 0, 0, 0)


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Build sanitized public Round185 run-level artifacts.")
    parser.add_argument("--archive-path", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--release-zip",
        type=Path,
        default=REPO_ROOT / "dist" / "round185_run_level_artifacts_sanitized.zip",
    )
    parser.add_argument("--check-only", action="store_true")
    return parser


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def read_json_entry(archive: zipfile.ZipFile, name: str) -> Any:
    return json.loads(archive.read(name).decode("utf-8-sig"))


def portable_path(value: str) -> str:
    normalized = value.replace("\\", "/")
    lowered = normalized.lower()
    for marker, placeholder in (
        ("/cloud_batch_round185/", "<BATCH_ROOT>/"),
        ("/zero-violation-paper-workbench/", "<REPO_ROOT>/"),
    ):
        position = lowered.find(marker)
        if position >= 0:
            return placeholder + normalized[position + len(marker) :]
    if re.match(r"(?i)^[A-Z]:/", normalized):
        return "<HOST_PATH>/" + normalized.rsplit("/", 1)[-1]
    return value


def sanitize_text(text: str) -> str:
    normalized = text.replace("\r\r\n", "\n").replace("\r\n", "\n").replace("\r", "\n")
    return WINDOWS_PATH_RE.sub(lambda match: portable_path(match.group(0)), normalized)


def sanitize_json(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key in DROP_HOST_KEYS:
                continue
            sanitized[key] = sanitize_json(item)
        return sanitized
    if isinstance(value, list):
        return [sanitize_json(item) for item in value]
    if isinstance(value, str):
        return sanitize_text(value)
    return value


def archive_run_ids(archive: zipfile.ZipFile) -> list[str]:
    pattern = re.compile(r"^runs/([^/]+)/result\.json$")
    return sorted(match.group(1) for name in archive.namelist() if (match := pattern.match(name)))


def single_entry(archive: zipfile.ZipFile, pattern: str) -> str:
    matches = sorted(name for name in archive.namelist() if re.match(pattern, name))
    if len(matches) != 1:
        raise ValueError(f"Expected one archive entry for {pattern!r}, found {len(matches)}")
    return matches[0]


def method_result(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    results = payload.get("results", {})
    if len(results) != 1:
        raise ValueError(f"Expected one method result, found {sorted(results)}")
    method, result = next(iter(results.items()))
    return str(method), result


def public_run_spec(raw_spec: dict[str, Any]) -> dict[str, Any]:
    payload = {key: deepcopy(value) for key, value in raw_spec.items() if key not in RUN_PATH_KEYS}
    payload["artifacts"] = {
        "config": "config.json",
        "evaluation": "evaluation.json",
        "progress": "progress.csv",
    }
    return sanitize_json(payload)


def public_evaluation(run_id: str, payload: dict[str, Any]) -> dict[str, Any]:
    method, result = method_result(payload)
    evaluation = result.get("evaluation", {})
    summary = deepcopy(evaluation.get("summary", {}))
    return {
        "actual_steps": result.get("progress", {}).get("total_env_steps"),
        "checkpoint_selection": "latest_saved_checkpoint",
        "env_id": payload.get("env_id"),
        "evaluation_status": result.get("evaluation_status"),
        "framework": result.get("framework"),
        "framework_version": result.get("framework_version"),
        "method": method,
        "policy": "deterministic",
        "requested_steps": result.get("requested_steps"),
        "run_id": run_id,
        "seed": payload.get("seed"),
        "summary": summary,
    }


def public_metric_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    public_rows = sanitize_json(deepcopy(rows))
    for row in public_rows:
        row.pop("checkpoint", None)
        row["checkpoint_selection"] = "latest_saved_checkpoint"
    return public_rows


def validate_matrix(run_specs: list[dict[str, Any]]) -> None:
    observed = {(item["method"], item["env_id"], int(item["seed"])) for item in run_specs}
    expected = {
        (method, env_id, seed)
        for method in EXPECTED_METHODS
        for env_id in EXPECTED_ENVS
        for seed in EXPECTED_SEEDS
    }
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise ValueError(f"Round185 matrix mismatch: missing={missing}, extra={extra}")


def safe_reset_runs_root(repo_root: Path) -> Path:
    data_root = (repo_root / "data").resolve()
    runs_root = (data_root / "runs").resolve()
    if runs_root.parent != data_root or runs_root.name != "runs":
        raise ValueError(f"Unsafe public runs path: {runs_root}")
    if runs_root.exists():
        shutil.rmtree(runs_root)
    runs_root.mkdir(parents=True)
    return runs_root


def write_public_data(archive_path: Path, repo_root: Path) -> dict[str, Any]:
    source_hash = file_sha256(archive_path)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected Round185 source SHA-256: {source_hash}")

    runs_root = safe_reset_runs_root(repo_root)
    manifest_runs: list[dict[str, Any]] = []
    run_specs: list[dict[str, Any]] = []
    episode_count = 0

    with zipfile.ZipFile(archive_path) as archive:
        run_ids = archive_run_ids(archive)
        if len(run_ids) != EXPECTED_RUNS:
            raise ValueError(f"Expected {EXPECTED_RUNS} result files, found {len(run_ids)}")

        for run_id in run_ids:
            prefix = f"runs/{run_id}/"
            result_payload = read_json_entry(archive, prefix + "result.json")
            raw_spec = read_json_entry(archive, prefix + "run_spec.json")
            config_name = single_entry(archive, rf"^{re.escape(prefix)}train/.+/config\.json$")
            progress_name = single_entry(archive, rf"^{re.escape(prefix)}train/.+/progress\.csv$")
            config_payload = sanitize_json(read_json_entry(archive, config_name))
            run_spec = public_run_spec(raw_spec)
            evaluation = public_evaluation(run_id, result_payload)
            summary = evaluation["summary"]
            for field in ("episode_returns", "episode_costs", "episode_lengths"):
                if len(summary.get(field, [])) != EXPECTED_EPISODES_PER_RUN:
                    raise ValueError(f"{run_id}: expected 50 values for {field}")
            episode_count += len(summary["episode_costs"])

            run_root = runs_root / run_id
            run_root.mkdir()
            (run_root / "config.json").write_bytes(json_bytes(config_payload))
            (run_root / "run_spec.json").write_bytes(json_bytes(run_spec))
            (run_root / "evaluation.json").write_bytes(json_bytes(evaluation))
            progress = sanitize_text(archive.read(progress_name).decode("utf-8-sig")).encode("utf-8")
            (run_root / "progress.csv").write_bytes(progress)

            run_specs.append(run_spec)
            manifest_runs.append(
                {
                    "env_id": run_spec["env_id"],
                    "files": {name: f"runs/{run_id}/{name}" for name in REQUIRED_RUN_FILES},
                    "method": run_spec["method"],
                    "run_id": run_id,
                    "seed": run_spec["seed"],
                }
            )

        validate_matrix(run_specs)
        metric_rows = public_metric_rows(read_json_entry(archive, "summaries/metric_table.json"))
        if len(metric_rows) != EXPECTED_RUNS:
            raise ValueError(f"Expected {EXPECTED_RUNS} metric rows, found {len(metric_rows)}")
        (repo_root / "data" / "metric_table.json").write_bytes(json_bytes(metric_rows))

    manifest = {
        "artifact_counts": {
            "config_files": EXPECTED_RUNS,
            "episode_evaluations": episode_count,
            "evaluation_files": EXPECTED_RUNS,
            "progress_files": EXPECTED_RUNS,
            "run_specs": EXPECTED_RUNS,
        },
        "matrix": {
            "environments": list(EXPECTED_ENVS),
            "methods": list(EXPECTED_METHODS),
            "runs": EXPECTED_RUNS,
            "seeds": list(EXPECTED_SEEDS),
            "training_steps_per_run": 5_000_000,
        },
        "runs": manifest_runs,
        "schema_version": 1,
        "source_archive_sha256": source_hash,
    }
    (repo_root / "data" / "run_manifest.json").write_bytes(json_bytes(manifest))
    return manifest


def sanitized_entry_bytes(entry: zipfile.ZipInfo, content: bytes) -> bytes:
    suffix = Path(entry.filename).suffix.lower()
    if suffix == ".json":
        return json_bytes(sanitize_json(json.loads(content.decode("utf-8-sig"))))
    if suffix in TEXT_EXTENSIONS:
        return sanitize_text(content.decode("utf-8", errors="replace")).encode("utf-8")
    return content


def write_zip_entry(archive: zipfile.ZipFile, name: str, content: bytes) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content, compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def write_release_zip(source_path: Path, output_path: Path) -> dict[str, Any]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source_path) as source, zipfile.ZipFile(output_path, "w") as output:
        entries = sorted((entry for entry in source.infolist() if not entry.is_dir()), key=lambda item: item.filename)
        for entry in entries:
            write_zip_entry(output, entry.filename, sanitized_entry_bytes(entry, source.read(entry)))
    digest = file_sha256(output_path)
    checksum_path = output_path.with_name("SHA256SUMS.txt")
    checksum_path.write_text(f"{digest}  {output_path.name}\n", encoding="ascii")
    return {"entries": len(entries), "sha256": digest, "size_bytes": output_path.stat().st_size}


def scan_private_text(root: Path) -> list[str]:
    hits: list[str] = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.suffix.lower() not in TEXT_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if PRIVATE_TEXT_RE.search(text):
            hits.append(str(path.relative_to(root)))
    return hits


def verify_release_zip(path: Path) -> dict[str, Any]:
    hits: list[str] = []
    with zipfile.ZipFile(path) as archive:
        entries = [entry for entry in archive.infolist() if not entry.is_dir()]
        for entry in entries:
            if Path(entry.filename).suffix.lower() not in TEXT_EXTENSIONS:
                continue
            text = archive.read(entry).decode("utf-8", errors="replace")
            if PRIVATE_TEXT_RE.search(text):
                hits.append(entry.filename)
    if hits:
        raise ValueError(f"Private text remains in release archive: {hits[:10]}")
    return {"entries": len(entries), "private_text_hits": len(hits)}


def verify_public_data(archive_path: Path, repo_root: Path, release_zip: Path) -> dict[str, Any]:
    runs_root = repo_root / "data" / "runs"
    run_roots = sorted(item for item in runs_root.iterdir() if item.is_dir())
    if len(run_roots) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} public run directories, found {len(run_roots)}")

    episode_count = 0
    with zipfile.ZipFile(archive_path) as archive:
        source_metric_rows = public_metric_rows(read_json_entry(archive, "summaries/metric_table.json"))
        published_metric_rows = json.loads((repo_root / "data" / "metric_table.json").read_text(encoding="utf-8"))
        if published_metric_rows != source_metric_rows:
            raise ValueError("Public metric table differs from the sanitized source matrix")
        for run_root in run_roots:
            run_id = run_root.name
            for name in REQUIRED_RUN_FILES:
                if not (run_root / name).is_file():
                    raise ValueError(f"Missing {run_id}/{name}")
            public_eval = json.loads((run_root / "evaluation.json").read_text(encoding="utf-8"))
            source_result = read_json_entry(archive, f"runs/{run_id}/result.json")
            expected_eval = public_evaluation(run_id, source_result)
            if public_eval != expected_eval:
                raise ValueError(f"Evaluation data changed for {run_id}")
            episode_count += len(public_eval["summary"]["episode_costs"])

            progress_name = single_entry(archive, rf"^runs/{re.escape(run_id)}/train/.+/progress\.csv$")
            expected_progress = sanitize_text(archive.read(progress_name).decode("utf-8-sig")).encode("utf-8")
            if (run_root / "progress.csv").read_bytes() != expected_progress:
                raise ValueError(f"Training curve changed for {run_id}")

    if episode_count != EXPECTED_EPISODES:
        raise ValueError(f"Expected {EXPECTED_EPISODES} public episodes, found {episode_count}")
    private_hits = scan_private_text(repo_root / "data")
    if private_hits:
        raise ValueError(f"Private text remains in public data: {private_hits[:10]}")
    release = verify_release_zip(release_zip)
    return {
        "episode_count": episode_count,
        "private_text_hits": len(private_hits),
        "release_entries": release["entries"],
        "run_count": len(run_roots),
    }


def resolve_args(args: Namespace) -> Namespace:
    args.archive_path = args.archive_path.resolve()
    args.repo_root = args.repo_root.resolve()
    args.release_zip = args.release_zip.resolve()
    return args


def main(argv: list[str] | None = None) -> int:
    args = resolve_args(build_parser().parse_args(argv))
    source_hash = file_sha256(args.archive_path)
    if source_hash != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"Unexpected Round185 source SHA-256: {source_hash}")

    if not args.check_only:
        manifest = write_public_data(args.archive_path, args.repo_root)
        release = write_release_zip(args.archive_path, args.release_zip)
        print(f"public_manifest_runs={manifest['matrix']['runs']}")
        print(f"release_entries={release['entries']}")
        print(f"release_size_bytes={release['size_bytes']}")
        print(f"release_sha256={release['sha256']}")

    verification = verify_public_data(args.archive_path, args.repo_root, args.release_zip)
    for key, value in verification.items():
        print(f"verified_{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
