#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from sweagent.run.batch_instances import SWEBenchInstances
from swerex.deployment.config import DockerDeploymentConfig


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--subset", default="full")
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-format", choices=["text", "json"], default="text")
    args = parser.parse_args()

    source = SWEBenchInstances(
        subset=args.subset,
        split=args.split,
        filter=f"^{args.instance_id}$",
        deployment=DockerDeploymentConfig(image="python:3.11"),
    )
    items = source.get_instance_configs()
    if len(items) != 1:
        raise SystemExit(f"expected 1 instance for {args.instance_id}, got {len(items)}")

    image = items[0].env.deployment.image
    if args.output_format == "json":
        print(json.dumps({"instance_id": args.instance_id, "image": image}, ensure_ascii=False))
    else:
        print(image)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
