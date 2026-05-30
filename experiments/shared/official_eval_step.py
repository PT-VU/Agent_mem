#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def _run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_report(workspace_root: Path, attempt_dir: Path, eval_run_id: str) -> Path | None:
    candidates = [
        attempt_dir / f"preds.{eval_run_id}.json",
        attempt_dir / f"attempt_01.{eval_run_id}.json",
        workspace_root / f"preds.{eval_run_id}.json",
        workspace_root / f"attempt_01.{eval_run_id}.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    found = list(attempt_dir.glob(f"*{eval_run_id}*.json")) + list(workspace_root.glob(f"*{eval_run_id}*.json"))
    return found[0] if found else None


def main() -> int:
    ap = argparse.ArgumentParser(description="Run official eval for one attempt output and optionally apply feedback.")
    ap.add_argument("--workspace-root", default="/home/pt/SWE-bench")
    ap.add_argument("--python-bin", default=sys.executable)
    ap.add_argument("--instance-id", required=True)
    ap.add_argument("--attempt-dir", required=True)
    ap.add_argument("--predictions-json", required=True)
    ap.add_argument("--report-json", required=True)
    ap.add_argument("--summary-json", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--scope", required=True)
    ap.add_argument("--dataset", default="SWE-bench/SWE-bench test")
    ap.add_argument("--split", default="test")
    ap.add_argument("--eval-timeout-sec", type=int, default=1800)
    ap.add_argument("--max-workers-eval", type=int, default=1)
    ap.add_argument("--feedback-json", default="")
    ap.add_argument("--apply-feedback", action="store_true")
    ap.add_argument("--config", default="")
    args = ap.parse_args()

    workspace_root = Path(args.workspace_root).resolve()
    attempt_dir = Path(args.attempt_dir).resolve()
    predictions_json = Path(args.predictions_json).resolve()
    report_json = Path(args.report_json).resolve()
    summary_json = Path(args.summary_json).resolve()
    feedback_json = Path(args.feedback_json).resolve() if args.feedback_json else None

    common_dir = Path(__file__).resolve().parent / "common"
    rebuild_script = common_dir / "rebuild_eval_predictions.py"
    summarize_script = common_dir / "summarize_official_eval.py"
    stub_script = common_dir / "write_local_eval_stub.py"
    feedback_script = common_dir / "apply_official_eval_feedback.py"

    with tempfile.TemporaryDirectory(prefix="phase7_eval_") as tmpdir:
        cache_path = Path(tmpdir) / "instance_ids.json"
        cache_path.write_text(json.dumps([args.instance_id], ensure_ascii=False, indent=2), encoding="utf-8")

        predictions_json.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                args.python_bin,
                str(rebuild_script),
                "--output-dir",
                str(attempt_dir),
                "--cache-file",
                str(cache_path),
                "--default-model-name",
                "phase7_final_validation",
                "--out",
                str(predictions_json),
            ]
        )
        pred_data = _load_json(predictions_json) if predictions_json.exists() else {}
        if not isinstance(pred_data, dict) or not pred_data:
            report_json.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    args.python_bin,
                    str(stub_script),
                    "--instance-id",
                    args.instance_id,
                    "--outcome",
                    "incomplete",
                    "--reason",
                    "missing_prediction",
                    "--out",
                    str(report_json),
                ]
            )
        else:
            _run(
                [
                    args.python_bin,
                    "-m",
                    "swebench.harness.run_evaluation",
                    "--dataset_name",
                    "SWE-bench/SWE-bench",
                    "--split",
                    args.split,
                    "--predictions_path",
                    str(predictions_json),
                    "--max_workers",
                    str(args.max_workers_eval),
                    "--timeout",
                    str(args.eval_timeout_sec),
                    "--cache_level",
                    "env",
                    "--clean",
                    "false",
                    "--run_id",
                    args.run_id,
                    "--instance_ids",
                    args.instance_id,
                ],
                cwd=attempt_dir,
            )
            latest_report = _resolve_report(workspace_root, attempt_dir, args.run_id)
            if latest_report is None:
                raise SystemExit(f"failed to resolve official eval report for run_id={args.run_id}")
            report_json.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(latest_report, report_json)

        summary_json.parent.mkdir(parents=True, exist_ok=True)
        _run(
            [
                args.python_bin,
                str(summarize_script),
                "--report-json",
                str(report_json),
                "--run-id",
                args.run_id,
                "--dataset",
                args.dataset,
                "--scope",
                args.scope,
                "--summary-out",
                str(summary_json),
            ]
        )

        if args.apply_feedback:
            if feedback_json is None:
                raise SystemExit("--feedback-json is required when --apply-feedback is set")
            feedback_json.parent.mkdir(parents=True, exist_ok=True)
            _run(
                [
                    args.python_bin,
                    str(feedback_script),
                    "--workspace-root",
                    str(workspace_root),
                    "--report-json",
                    str(report_json),
                    "--output-dir",
                    str(attempt_dir),
                    "--cache-file",
                    str(cache_path),
                    "--run-id",
                    args.run_id,
                    "--output-json",
                    str(feedback_json),
                    *([] if not args.config else ["--config", args.config]),
                ]
            )

    print(json.dumps({"report_json": str(report_json), "summary_json": str(summary_json), "feedback_json": str(feedback_json) if feedback_json else ""}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
