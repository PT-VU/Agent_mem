#!/usr/bin/env python3
"""Rebuild SWE-bench evaluation predictions from run outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_patch_from_pred(pred_path: Path) -> tuple[str, str]:
    """
    Return (model_name_or_path, model_patch) from a .pred file.
    .pred can be:
    1) a JSON object with keys model_name_or_path / model_patch
    2) a raw git patch text
    """
    text = pred_path.read_text(encoding="utf-8", errors="ignore")
    stripped = text.lstrip()
    if not stripped:
        return "", ""

    if stripped.startswith("{"):
        try:
            obj = json.loads(stripped)
            if isinstance(obj, dict):
                model_name = str(obj.get("model_name_or_path", "") or "")
                patch = str(obj.get("model_patch", "") or "")
                return model_name, patch
        except Exception:
            pass

    # Fallback: treat as raw patch.
    return "", text


def _extract_patch_from_traj(traj_path: Path) -> tuple[str, str]:
    try:
        traj = _read_json(traj_path)
    except Exception:
        return "", ""
    info = traj.get("info", {}) if isinstance(traj, dict) else {}
    patch = str(info.get("submission", "") or "")
    return "", patch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, help="run_outputs directory")
    parser.add_argument("--out", required=True, help="output predictions json path")
    parser.add_argument(
        "--default-model-name",
        default="run_outputs",
        help="fallback model_name_or_path if not found in .pred",
    )
    parser.add_argument(
        "--cache-file",
        default=None,
        help="optional cache JSON (list of instance ids) for deterministic ordering",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.cache_file:
        instance_ids = list(_read_json(Path(args.cache_file).resolve()))
    else:
        instance_ids = sorted([p.name for p in outdir.iterdir() if p.is_dir()])

    predictions: dict[str, dict] = {}
    stats = {
        "requested_instance_ids": len(instance_ids),
        "with_patch": 0,
        "from_pred_json": 0,
        "from_pred_raw": 0,
        "from_traj_fallback": 0,
        "missing_patch": 0,
        "non_diff_prefix": 0,
    }

    for instance_id in instance_ids:
        inst_dir = outdir / instance_id
        pred_path = inst_dir / f"{instance_id}.pred"
        traj_path = inst_dir / f"{instance_id}.traj"

        model_name = args.default_model_name
        patch = ""
        source = ""

        if pred_path.exists():
            pred_model, pred_patch = _extract_patch_from_pred(pred_path)
            if pred_model:
                model_name = pred_model
            if pred_patch:
                patch = pred_patch
                source = "pred_json" if pred_path.read_text(encoding="utf-8", errors="ignore").lstrip().startswith("{") else "pred_raw"

        if not patch and traj_path.exists():
            traj_model, traj_patch = _extract_patch_from_traj(traj_path)
            if traj_model:
                model_name = traj_model
            if traj_patch:
                patch = traj_patch
                source = "traj"

        if not patch.strip():
            stats["missing_patch"] += 1
            continue

        if not patch.endswith("\n"):
            patch += "\n"

        if not patch.lstrip().startswith("diff --git "):
            stats["non_diff_prefix"] += 1

        if source == "pred_json":
            stats["from_pred_json"] += 1
        elif source == "pred_raw":
            stats["from_pred_raw"] += 1
        elif source == "traj":
            stats["from_traj_fallback"] += 1

        predictions[instance_id] = {
            "model_name_or_path": model_name,
            "instance_id": instance_id,
            "model_patch": patch,
        }

    stats["with_patch"] = len(predictions)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(predictions, f, ensure_ascii=False, indent=2)

    print(f"wrote: {out_path}")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
