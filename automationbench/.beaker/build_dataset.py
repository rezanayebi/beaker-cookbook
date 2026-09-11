"""Build and upload a Beaker dataset from the recipe's frozen splits.

Rows name tasks the way ``splits/{train,test}.txt`` do — by ``task_name`` — so
the dataset carries no copy of the benchmark's prompts, simulated state, or
assertions; those come from the pinned ``automation-bench`` dependency at run
time. Nothing is written into the repository: the JSONL is staged in an
OS-managed temporary directory and uploaded before it is cleaned up.

    uv run python .beaker/build_dataset.py --name <dataset> --train 10 --test 5

``--train``/``--test`` take a deterministic domain-balanced slice of each frozen
split (round-robin over the six domains, preserving each split file's own
order); omit them to upload the whole split.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def rows_for(split: str, limit: int | None) -> list[dict[str, Any]]:
    from automationbench_skills.data.tasks import load_split, task_family

    samples = load_split(split)
    if limit is not None:
        samples = balanced_slice(samples, limit)
    return [
        {
            "task_name": sample.task_name,
            "domain": sample.domain,
            "task_family": task_family(sample.task_name),
        }
        for sample in samples
    ]


def balanced_slice(samples: list[Any], limit: int) -> list[Any]:
    """Take ``limit`` samples round-robin across domains, in split-file order."""
    by_domain: dict[str, list[Any]] = {}
    for sample in samples:
        by_domain.setdefault(sample.domain, []).append(sample)
    picked: list[Any] = []
    position = 0
    while len(picked) < limit and any(len(queue) > position for queue in by_domain.values()):
        for domain in sorted(by_domain):
            if len(picked) == limit:
                break
            queue = by_domain[domain]
            if len(queue) > position:
                picked.append(queue[position])
        position += 1
    # Report in the frozen split's own order, not the round-robin order.
    order = {sample.task_name: i for i, sample in enumerate(samples)}
    return sorted(picked, key=lambda s: order[s.task_name])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Human-readable dataset name.")
    parser.add_argument("--train", type=int, default=None, help="Rows to take from splits/train.txt.")
    parser.add_argument("--test", type=int, default=None, help="Rows to take from splits/test.txt.")
    parser.add_argument("--agent", default=None, help="Beaker agent key; defaults to the configured one.")
    args = parser.parse_args()

    splits = {"train": rows_for("train", args.train), "test": rows_for("test", args.test)}
    for name, rows in splits.items():
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["domain"]] = counts.get(row["domain"], 0) + 1
        print(f"{name}: {len(rows)} rows {dict(sorted(counts.items()))}", file=sys.stderr)

    with tempfile.TemporaryDirectory(prefix="beaker-automationbench-") as temp_dir:
        dataset_dir = Path(temp_dir)
        for name, rows in splits.items():
            with (dataset_dir / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")

        command = [
            "beaker",
            "dataset",
            "upload",
            str(dataset_dir),
            "--name",
            args.name,
            "--total-count",
            str(sum(len(rows) for rows in splits.values())),
            *[arg for name, rows in splits.items() for arg in ("--split", f"{name}={len(rows)}")],
            "--metadata",
            "source=automationbench frozen splits",
            "--metadata",
            f"benchmark_pin={benchmark_pin()}",
            "--json",
        ]
        if args.agent:
            command += ["--agent", args.agent]
        upload = subprocess.run(command, check=True, capture_output=True, text=True, cwd=PROJECT_ROOT)

    artifact = json.loads(upload.stdout)
    print(f"{artifact['artifact_key']}@{artifact['dataset_revision']}")
    return 0


def benchmark_pin() -> str:
    """The automation-bench commit the rows are valid against."""
    import importlib.metadata as metadata

    try:
        return metadata.version("automation-bench")
    except metadata.PackageNotFoundError:  # pragma: no cover - dependency always present
        return "unknown"


if __name__ == "__main__":
    raise SystemExit(main())
