from __future__ import annotations

from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_public_run_data import (  # noqa: E402
    DROP_HOST_KEYS,
    PRIVATE_TEXT_RE,
    public_metric_rows,
    public_run_spec,
    sanitized_entry_bytes,
    sanitize_json,
    sanitize_text,
)
import zipfile


class PublicRunDataTests(unittest.TestCase):
    def test_sanitize_json_removes_host_fields_and_rewrites_paths(self) -> None:
        host_python = "C:" + r"\Users\cloud\miniconda3\python.exe"
        repo_root = "D:" + r"\work\zero-violation-paper-workbench"
        run_log = repo_root + r"\experiments\results\external_saferl\cloud_batch_round185\runs\x\train"
        payload = {
            "python_executable": host_python,
            "repo_root": repo_root,
            "nested": {"log_dir": run_log},
        }

        sanitized = sanitize_json(payload)

        self.assertTrue(DROP_HOST_KEYS.isdisjoint(sanitized))
        self.assertEqual(sanitized["nested"]["log_dir"], "<BATCH_ROOT>/runs/x/train")

    def test_sanitize_text_removes_windows_paths(self) -> None:
        text = "checkpoint=D:" + r"\batch\cloud_batch_round185\runs\x\epoch-250.pt"
        sanitized = sanitize_text(text)

        self.assertEqual(sanitized, "checkpoint=<BATCH_ROOT>/runs/x/epoch-250.pt")
        self.assertIsNone(PRIVATE_TEXT_RE.search(sanitized))
        self.assertIsNone(PRIVATE_TEXT_RE.search("https://github.com/openai/example"))

    def test_sanitize_text_normalizes_duplicate_carriage_returns(self) -> None:
        self.assertEqual(sanitize_text("a\r\r\nb\r\n"), "a\nb\n")

    def test_public_run_spec_replaces_artifact_paths(self) -> None:
        run_dir = "D:" + r"\batch\cloud_batch_round185\runs\x"
        raw = {
            "run_id": "ppo__SafetyPointGoal1-v0__seed001",
            "method": "PPO",
            "env_id": "SafetyPointGoal1-v0",
            "seed": 1,
            "run_dir": run_dir,
            "output_json": run_dir + r"\result.json",
        }

        public = public_run_spec(raw)

        self.assertNotIn("run_dir", public)
        self.assertNotIn("output_json", public)
        self.assertEqual(public["artifacts"]["evaluation"], "evaluation.json")

    def test_json_entry_accepts_utf8_bom(self) -> None:
        entry = zipfile.ZipInfo("manifest.json")

        content = sanitized_entry_bytes(entry, b'\xef\xbb\xbf{"repo_root": "D:\\\\work"}')

        self.assertEqual(content, b"{}\n")

    def test_public_metric_rows_records_selection_without_path(self) -> None:
        rows = [{"run_id": "x", "checkpoint": "C:" + r"\batch\epoch-250.pt", "return": 1.0}]

        public = public_metric_rows(rows)

        self.assertNotIn("checkpoint", public[0])
        self.assertEqual(public[0]["checkpoint_selection"], "latest_saved_checkpoint")
        self.assertEqual(public[0]["return"], 1.0)


if __name__ == "__main__":
    unittest.main()
