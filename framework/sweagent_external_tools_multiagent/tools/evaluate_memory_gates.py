from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, int]:
    out = {
        "plan_generated": 0,
        "external_tool_response": 0,
        "memory_hint_buffered": 0,
        "memory_injected": 0,
        "action_error_events": 0,
        "error_like_observation": 0,
    }
    family_counter: Counter[str] = Counter()
    repeated_family_injections = 0
    for row in rows:
        tool = row.get("tool")
        payload = row.get("payload", {})
        if not isinstance(payload, dict):
            continue
        event = str(payload.get("event", ""))
        if event == "memory_gate_evaluation" and isinstance(payload.get("metrics"), dict):
            # Prefer direct gate summary when present.
            metrics = payload["metrics"]
            for key in out:
                if isinstance(metrics.get(key), int):
                    out[key] = max(out[key], int(metrics[key]))
            continue
        if event == "plan_generated":
            out["plan_generated"] += 1
        if tool == "HOOK" and event == "external_tool_response":
            out["external_tool_response"] += 1
        if tool == "HOOK" and event == "memory_hint_buffered":
            out["memory_hint_buffered"] += int(payload.get("buffered_count", 1) or 1)
        if tool == "HOOK" and event == "memory_injected":
            out["memory_injected"] += int(payload.get("hint_count", 0) or 0)
            for raw_family in payload.get("selected_family_ids", []) or []:
                family = str(raw_family or "").strip()
                if not family:
                    continue
                if family_counter[family] > 0:
                    repeated_family_injections += 1
                family_counter[family] += 1
        if event == "action_error":
            out["action_error_events"] += 1
    out["family_injection_total"] = int(sum(family_counter.values()))
    out["repeated_family_injections"] = repeated_family_injections
    out["unique_family_injections"] = len(family_counter)
    return out


def _evaluate(
    metrics: dict[str, int],
    *,
    min_external_ratio: float,
    min_buffer_ratio: float,
    min_action_error_coverage: float,
) -> dict[str, Any]:
    pg = max(1, metrics["plan_generated"])
    et = metrics["external_tool_response"]
    hb = metrics["memory_hint_buffered"]
    ae = metrics["action_error_events"]
    eo = metrics["error_like_observation"]
    external_ratio = et / pg
    buffer_ratio = (hb / et) if et > 0 else 0.0
    action_error_coverage = (ae / eo) if eo > 0 else 1.0
    gates = {
        "external_ratio": external_ratio >= min_external_ratio,
        "buffer_ratio": buffer_ratio >= min_buffer_ratio,
        "action_error_coverage": action_error_coverage >= min_action_error_coverage,
    }
    return {
        "metrics": metrics,
        "ratios": {
            "external_tool_response_over_plan_generated": round(external_ratio, 6),
            "memory_hint_buffered_over_external_tool_response": round(buffer_ratio, 6),
            "action_error_over_error_like_observation": round(action_error_coverage, 6),
            "repeated_family_injection_rate": round(
                _safe_div(metrics.get("repeated_family_injections", 0), metrics.get("family_injection_total", 0)),
                6,
            ),
        },
        "thresholds": {
            "external_ratio": min_external_ratio,
            "buffer_ratio": min_buffer_ratio,
            "action_error_coverage": min_action_error_coverage,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }


def _safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return float(num / den)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate AgentMem extraction/injection gate metrics.")
    parser.add_argument("--log-file", required=True)
    parser.add_argument("--output", default="")
    parser.add_argument("--min-external-ratio", type=float, default=0.35)
    parser.add_argument("--min-buffer-ratio", type=float, default=0.15)
    parser.add_argument("--min-action-error-coverage", type=float, default=0.60)
    parser.add_argument("--hard-fail", action="store_true")
    args = parser.parse_args()

    rows = _load_jsonl(Path(args.log_file))
    metrics = _aggregate(rows)
    report = _evaluate(
        metrics,
        min_external_ratio=args.min_external_ratio,
        min_buffer_ratio=args.min_buffer_ratio,
        min_action_error_coverage=args.min_action_error_coverage,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n", encoding="utf-8")
    if args.hard_fail and not report["passed"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
