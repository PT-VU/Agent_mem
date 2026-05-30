from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_INFRA_MARKERS: tuple[tuple[str, str], ...] = (
    ("insufficient balance", "llm_insufficient_balance"),
    ("dockerpullerror", "docker_pull_error"),
    ("docker pull", "docker_pull_error"),
    ("docker build", "docker_build_error"),
    ("daemon unavailable", "docker_daemon_unavailable"),
    ("docker daemon", "docker_daemon_unavailable"),
    ("no space left on device", "disk_full"),
    ("sigbus", "docker_sigbus"),
)


def build_outcome_map(report: dict[str, Any]) -> dict[str, str]:
    outcome_map: dict[str, str] = {}
    for key, outcome in (
        ("resolved_ids", "resolved"),
        ("unresolved_ids", "unresolved"),
        ("incomplete_ids", "incomplete"),
        ("error_ids", "incomplete"),
    ):
        values = report.get(key)
        if not isinstance(values, list):
            continue
        for instance_id in values:
            text = str(instance_id).strip()
            if text:
                outcome_map[text] = outcome
    return outcome_map


def load_traj_payload(instance_dir: Path) -> dict[str, Any]:
    traj_path = instance_dir / f"{instance_dir.name}.traj"
    if not traj_path.exists():
        return {}
    try:
        payload = json.loads(traj_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def load_patch_text(instance_dir: Path) -> str:
    pred_path = instance_dir / f"{instance_dir.name}.pred"
    if pred_path.exists():
        text = pred_path.read_text(encoding="utf-8", errors="ignore")
        stripped = text.lstrip()
        if stripped.startswith("{"):
            try:
                obj = json.loads(stripped)
            except Exception:
                obj = None
            if isinstance(obj, dict):
                patch = str(obj.get("model_patch") or "").strip()
                if patch:
                    return patch + ("\n" if not patch.endswith("\n") else "")
        if stripped:
            return text if text.endswith("\n") else text + "\n"

    patch_path = instance_dir / f"{instance_dir.name}.patch"
    if patch_path.exists():
        text = patch_path.read_text(encoding="utf-8", errors="ignore")
        if text.strip():
            return text if text.endswith("\n") else text + "\n"

    traj = load_traj_payload(instance_dir)
    info = traj.get("info") if isinstance(traj, dict) else {}
    if isinstance(info, dict):
        patch = str(info.get("submission") or "").strip()
        if patch:
            return patch + ("\n" if not patch.endswith("\n") else "")
    return ""


def extract_changed_files_from_patch(patch_text: str) -> list[str]:
    changed: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(r"(?m)^diff --git a/(.+?) b/(.+?)$", patch_text or ""):
        path = str(match.group(2)).strip()
        if not path or path in seen:
            continue
        seen.add(path)
        changed.append(path)
    return changed


def extract_validation_summary(instance_dir: Path) -> dict[str, Any]:
    traj = load_traj_payload(instance_dir)
    trajectory = traj.get("trajectory")
    if not isinstance(trajectory, list):
        return {"commands": []}

    commands: list[str] = []
    seen: set[str] = set()
    for step in trajectory:
        if not isinstance(step, dict):
            continue
        action = str(step.get("action") or "").strip()
        if not action:
            continue
        low = action.lower()
        if action.startswith("str_replace_editor"):
            continue
        if not any(token in low for token in ("pytest", "tox", "nox", "unittest", "reproduce", "python -m pytest", "python -m unittest")):
            continue
        if action in seen:
            continue
        seen.add(action)
        commands.append(action)
        if len(commands) >= 6:
            break
    return {"commands": commands}


def extract_task_summary(instance_dir: Path) -> str:
    traj = load_traj_payload(instance_dir)
    trajectory = traj.get("trajectory")
    if not isinstance(trajectory, list):
        return ""
    for step in trajectory[:6]:
        if not isinstance(step, dict):
            continue
        query = step.get("query")
        if not isinstance(query, list):
            continue
        for message in query:
            if not isinstance(message, dict):
                continue
            content = str(message.get("content") or "")
            match = re.search(r"<pr_description>\s*(.*?)\s*</pr_description>", content, re.S)
            if not match:
                continue
            text = re.sub(r"\s+", " ", match.group(1)).strip()
            if text:
                return text[:600]
    return ""


def infer_eval_outcome(base_outcome: str, instance_dir: Path) -> tuple[str, str]:
    normalized = str(base_outcome or "").strip().lower()
    if normalized in {"resolved", "unresolved"}:
        return normalized, ""

    text_parts: list[str] = []
    for suffix in (".trace.log", ".debug.log", ".info.log"):
        path = instance_dir / f"{instance_dir.name}{suffix}"
        if path.exists():
            text_parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    body = "\n".join(text_parts).lower()
    for token, reason in _INFRA_MARKERS:
        if token in body:
            return "infra_failure", reason
    if normalized == "resolved":
        return "resolved", ""
    return "incomplete", ""


def infer_attempt_id(run_id: str) -> str:
    text = str(run_id or "").strip()
    if not text:
        return ""
    match = re.search(r"_attempt_(\d+)(?:\b|$)", text)
    if not match:
        return ""
    return f"attempt-{match.group(1).zfill(2)}"


def build_feedback_event(
    *,
    instance_id: str,
    base_outcome: str,
    report_path: str,
    instance_dir: Path,
    run_id: str = "",
) -> dict[str, Any]:
    outcome, infra_reason = infer_eval_outcome(base_outcome, instance_dir)
    patch_text = load_patch_text(instance_dir)
    changed_files = extract_changed_files_from_patch(patch_text)
    validation_summary = extract_validation_summary(instance_dir)
    task_summary = extract_task_summary(instance_dir)
    attempt_id = infer_attempt_id(run_id)

    patch_summary = {
        "changed_file_count": len(changed_files),
        "risk_flags": [],
    }
    if outcome == "infra_failure" and infra_reason:
        patch_summary["infra_reason"] = infra_reason

    return {
        "event": "official_eval_feedback",
        "instance_id": instance_id,
        "official_eval_status": outcome,
        "outcome": outcome,
        "eval_ref": report_path,
        "report_path": report_path,
        "patch_text": patch_text,
        "patch_summary": patch_summary,
        "changed_files": changed_files,
        "validation_summary": validation_summary,
        "task_summary": task_summary,
        "run_id": run_id,
        "attempt_id": attempt_id,
    }
