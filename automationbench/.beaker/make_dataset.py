"""Derive the Beaker dataset from this repository's own labeled AutomationBench data.

Rows name tasks by ``task_name`` -- the same key the frozen ``splits/*.txt`` files
use -- and carry the labels the pinned ``automation-bench`` dependency defines
for them. The task prompt and its 15-80 KB simulated world are rebuilt from that
pin at rollout time rather than copied into the dataset.

Split mapping follows the repository's own leakage policy
(``src/automationbench_skills/splits/README.md``):

* ``train.jsonl``  -- from ``splits/train.txt``: the optimization pool.
* ``val.jsonl``    -- from ``splits/train.txt``, disjoint from train. Beaker
  requires this split and uses it to choose between candidates, and the policy
  says model selection happens on train, so it is drawn from train.
* ``test.jsonl``   -- from ``splits/test.txt``: held out for final evaluation,
  which is exactly when Beaker evaluates it (the winning candidate only).

Generated JSONL is written to an OS temporary directory and uploaded from there;
nothing is persisted in the repository. Run it as::

    uv run python .beaker/make_dataset.py --name <dataset-name>
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from automationbench_skills.data import PUBLIC_DOMAINS, Sample, load_split, task_family


# Snapshot file stem -> the split name the platform indexes. Beaker's loader
# reads the validation split from val.jsonl but names the split "validation";
# uploading it as "val" stores the file without presigning it for download.
SPLIT_NAMES = {"train": "train", "val": "validation", "test": "test"}


def _take_per_domain(
    samples: list[Sample], per_domain: int, extra: int = 0, exclude: set[str] | None = None
) -> list[Sample]:
    """Take the first ``per_domain`` tasks of each domain, in split-file order.

    The split files are already stratified across task families with a fixed
    seed, so taking a prefix is deterministic and family-diverse. ``extra`` adds
    that many more tasks, one per domain in ``PUBLIC_DOMAINS`` order.
    """
    exclude = exclude or set()
    by_domain: dict[str, list[Sample]] = defaultdict(list)
    for sample in samples:
        if sample.task_name not in exclude:
            by_domain[sample.domain].append(sample)

    chosen: list[Sample] = []
    for domain in PUBLIC_DOMAINS:
        available = by_domain.get(domain, [])
        if len(available) < per_domain:
            raise ValueError(f"domain {domain!r} has {len(available)} eligible task(s), need {per_domain}")
        chosen.extend(available[:per_domain])
    for offset in range(extra):
        domain = PUBLIC_DOMAINS[offset % len(PUBLIC_DOMAINS)]
        chosen.append(by_domain[domain][per_domain + offset // len(PUBLIC_DOMAINS)])
    return chosen


def _row(sample: Sample) -> dict[str, Any]:
    """One dataset row: the task id, its label, and reporting metadata."""
    assertions = list(sample.info.get("assertions") or [])
    if not assertions:
        raise ValueError(f"task {sample.task_name!r} has no assertions to score against")
    return {
        "id": sample.task_name,
        "input": {"task_name": sample.task_name},
        "expected": {
            "partial_credit": 1.0,
            "num_assertions": len(assertions),
            "assertions": assertions,
        },
        "metadata": {
            "domain": sample.domain,
            "family": task_family(sample.task_name),
            "benchmark_index": sample.index,
            "zapier_tools": list(sample.info.get("zapier_tools") or []),
        },
        "group_key": sample.domain,
    }


def build_splits(train_size: int, val_size: int, test_size: int) -> dict[str, list[dict[str, Any]]]:
    """Select the tasks for each split and render them as dataset rows."""
    train_pool = load_split("train")
    test_pool = load_split("test")

    per_domain_train, extra_train = divmod(train_size, len(PUBLIC_DOMAINS))
    per_domain_val, extra_val = divmod(val_size, len(PUBLIC_DOMAINS))
    per_domain_test, extra_test = divmod(test_size, len(PUBLIC_DOMAINS))

    train = _take_per_domain(train_pool, per_domain_train, extra_train)
    val = _take_per_domain(train_pool, per_domain_val, extra_val, exclude={s.task_name for s in train})
    test = _take_per_domain(test_pool, per_domain_test, extra_test)

    overlap = {s.task_name for s in train} & {s.task_name for s in val}
    if overlap:
        raise AssertionError(f"train and val overlap: {sorted(overlap)}")
    return {"train": [_row(s) for s in train], "val": [_row(s) for s in val], "test": [_row(s) for s in test]}


def _manifest(splits: dict[str, list[dict[str, Any]]], pin: str) -> dict[str, Any]:
    return {
        "source": "automationbench (zapier/AutomationBench), public scored domains",
        "labels": "deterministic assertions defined by the pinned dependency",
        "dependency_pin": pin,
        "selection": "first N tasks per domain in frozen split-file order",
        "split_sources": {
            "train": "src/automationbench_skills/splits/train.txt",
            "val": "src/automationbench_skills/splits/train.txt (disjoint from train)",
            "test": "src/automationbench_skills/splits/test.txt",
        },
        "split_counts": {name: len(rows) for name, rows in splits.items()},
    }


def _dependency_pin() -> str:
    text = (Path(__file__).resolve().parent.parent / "pyproject.toml").read_text()
    for line in text.splitlines():
        if "AutomationBench.git@" in line:
            return line.strip().strip('",')
    return "unknown"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="Hosted dataset name")
    parser.add_argument("--agent", default=None, help="Agent key (defaults to beaker.yaml agent_key)")
    parser.add_argument("--train-size", type=int, default=6)
    parser.add_argument("--val-size", type=int, default=6)
    parser.add_argument("--test-size", type=int, default=7)
    parser.add_argument("--dry-run", action="store_true", help="Print the selection without uploading")
    args = parser.parse_args()

    splits = build_splits(args.train_size, args.val_size, args.test_size)
    for name, rows in splits.items():
        print(f"{name}: {len(rows)} row(s)")
        for row in rows:
            print(f"  {row['id']}  ({row['expected']['num_assertions']} assertions)")
    if args.dry_run:
        return 0

    total = sum(len(rows) for rows in splits.values())
    with tempfile.TemporaryDirectory(prefix="beaker-dataset-") as temp_dir:
        dataset_dir = Path(temp_dir)
        for name, rows in splits.items():
            path = dataset_dir / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in rows:
                    handle.write(json.dumps(row) + "\n")
        (dataset_dir / "manifest.json").write_text(json.dumps(_manifest(splits, _dependency_pin()), indent=2))

        command = [
            "beaker",
            "dataset",
            "upload",
            str(dataset_dir),
            "--name",
            args.name,
            "--total-count",
            str(total),
            *[arg for name, rows in splits.items() for arg in ("--split", f"{SPLIT_NAMES[name]}={len(rows)}")],
            "--metadata",
            "source=automationbench-public",
            "--metadata",
            "labels=benchmark-assertions",
            "--json",
        ]
        if args.agent:
            command += ["--agent", args.agent]
        upload = subprocess.run(command, capture_output=True, text=True)
        if upload.returncode != 0:
            print(upload.stdout)
            print(upload.stderr)
            raise SystemExit(upload.returncode)

    artifact = json.loads(upload.stdout)
    print()
    print(f"artifact_id      : {artifact['artifact_id']}")
    print(f"dataset selector : {artifact['artifact_key']}@{artifact['dataset_revision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
