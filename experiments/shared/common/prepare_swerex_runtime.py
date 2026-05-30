#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass

from swerex.deployment.config import DockerDeploymentConfig
from swerex.deployment.docker import DockerDeployment


@dataclass
class WarmupResult:
    base_image: str
    built_image: str
    platform: str | None
    python_standalone_dir: str | None
    probe_ok: bool
    probe_output: str


def build_runtime_image(image: str, python_standalone_dir: str | None, platform: str | None) -> str:
    cfg = DockerDeploymentConfig(
        image=image,
        python_standalone_dir=python_standalone_dir,
        platform=platform,
        startup_timeout=600.0,
    )
    deployment = DockerDeployment.from_config(cfg)
    if not python_standalone_dir:
        return image
    return deployment._build_image()


def probe_runtime_image(image: str, python_standalone_dir: str | None, platform: str | None) -> tuple[bool, str]:
    if not python_standalone_dir:
        return True, "python_standalone_dir disabled; no probe executed"
    cmd = [
        "docker",
        "run",
        "--rm",
    ]
    if platform:
        cmd.extend(["--platform", platform])
    cmd.extend(
        [
            image,
            "/bin/sh",
            "-lc",
            f"{python_standalone_dir}/python3.11/bin/swerex-remote --version",
        ]
    )
    proc = subprocess.run(cmd, capture_output=True, text=True)
    output = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode == 0, output.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--python-standalone-dir", default="/root")
    parser.add_argument("--platform", default="linux/amd64")
    parser.add_argument("--output-format", choices=("text", "json"), default="text")
    args = parser.parse_args()

    standalone_dir = args.python_standalone_dir
    if standalone_dir in ("", "__NONE__", "none", "None"):
        standalone_dir = None

    built_image = build_runtime_image(args.image, standalone_dir, args.platform)
    probe_ok, probe_output = probe_runtime_image(built_image, standalone_dir, args.platform)
    result = WarmupResult(
        base_image=args.image,
        built_image=built_image,
        platform=args.platform,
        python_standalone_dir=standalone_dir,
        probe_ok=probe_ok,
        probe_output=probe_output,
    )
    if args.output_format == "json":
        print(json.dumps(asdict(result), ensure_ascii=False, indent=2))
    else:
        print(f"[warmup] base_image={result.base_image}")
        print(f"[warmup] built_image={result.built_image}")
        print(f"[warmup] platform={result.platform}")
        print(f"[warmup] python_standalone_dir={result.python_standalone_dir}")
        print(f"[warmup] probe_ok={result.probe_ok}")
        if result.probe_output:
            print("[warmup] probe_output_begin")
            print(result.probe_output)
            print("[warmup] probe_output_end")
    return 0 if probe_ok else 3


if __name__ == "__main__":
    raise SystemExit(main())
