"""Audit a protein manifest before embedding or large-scale training.

Checks exact sequence and declared cluster leakage across train/val/test. This
cannot establish structural-homology separation; use an external clustering
step before creating the manifest.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median


REQUIRED = ("id", "chain", "sequence", "split", "path")
SPLITS = ("train", "val", "test")


def audit(manifest_path, check_files=False):
    payload = json.loads(manifest_path.read_text())
    rows = payload["proteins"]
    if not isinstance(rows, list):
        raise ValueError("'proteins' must be a list")
    problems = []
    split_counts = Counter()
    lengths = defaultdict(list)
    sequence_splits = defaultdict(set)
    cluster_splits = defaultdict(set)
    seen_chains = set()
    missing_paths = []
    for number, row in enumerate(rows, 1):
        absent = [key for key in REQUIRED if key not in row]
        if absent:
            problems.append(f"row {number}: missing {', '.join(absent)}")
            continue
        split, sequence = row["split"], row["sequence"]
        if split not in SPLITS:
            problems.append(f"row {number}: invalid split {split!r}")
            continue
        if not isinstance(sequence, str) or not sequence:
            problems.append(f"row {number}: empty or invalid sequence")
            continue
        key = (str(row["id"]), str(row["chain"]))
        if key in seen_chains:
            problems.append(f"row {number}: duplicate chain {key}")
        seen_chains.add(key)
        if row.get("residues") is not None and row["residues"] != len(sequence):
            problems.append(f"row {number}: residues does not match sequence length")
        split_counts[split] += 1
        lengths[split].append(len(sequence))
        sequence_splits[sequence].add(split)
        if row.get("cluster") is not None:
            cluster_splits[str(row["cluster"])].add(split)
        if check_files:
            base = manifest_path.resolve().parent if payload.get("path_base") == "manifest" else (
                Path(__file__).resolve().parent.parent / "outputs")
            structure = (base / row["path"]).resolve()
            if not structure.is_file():
                missing_paths.append(str(structure))
    sequence_leaks = [dict(sequence=sequence, splits=sorted(splits))
                      for sequence, splits in sequence_splits.items() if len(splits) > 1]
    cluster_leaks = [dict(cluster=cluster, splits=sorted(splits))
                     for cluster, splits in cluster_splits.items() if len(splits) > 1]
    if sequence_leaks:
        problems.append(f"{len(sequence_leaks)} exact sequences occur across splits")
    if cluster_leaks:
        problems.append(f"{len(cluster_leaks)} declared clusters occur across splits")
    if missing_paths:
        problems.append(f"{len(missing_paths)} structure files are missing")
    if not split_counts["train"] or not split_counts["val"] or not split_counts["test"]:
        problems.append("train, val, and test must all be nonempty")
    return {
        "manifest": str(manifest_path.resolve()),
        "sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "rows": len(rows),
        "split_counts": {split: split_counts[split] for split in SPLITS},
        "lengths": {split: dict(min=min(lengths[split]), max=max(lengths[split]),
                              median=median(lengths[split])) if lengths[split] else None
                    for split in SPLITS},
        "cluster_coverage": sum(row.get("cluster") is not None for row in rows),
        "exact_sequence_leaks": sequence_leaks,
        "declared_cluster_leaks": cluster_leaks,
        "missing_structure_paths": missing_paths,
        "problems": problems,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--check-files", action="store_true")
    args = parser.parse_args()
    report = audit(args.manifest, args.check_files)
    encoded = json.dumps(report, indent=2)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(encoded + "\n")
    print(encoded)
    raise SystemExit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
