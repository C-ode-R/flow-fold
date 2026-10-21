"""Cluster AFDB candidates with experimental holdouts and freeze a train split.

Requires mmseqs2 on PATH. The existing holdouts are development references;
make a new untouched experimental test set for final scientific claims.
"""
import argparse
import json
import os
import random
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--holdout-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=16)
    args = parser.parse_args()
    if not shutil.which("mmseqs"):
        parser.error("mmseqs2 is required; install it before freezing the split")
    candidates = json.loads(args.candidates.read_text())["proteins"]
    holdouts = [row for row in json.loads(args.holdout_manifest.read_text())["proteins"]
                if row["split"] in ("val", "test") and 40 <= len(row["sequence"]) <= 128]
    if not all(any(row["split"] == split for row in holdouts) for split in ("val", "test")):
        parser.error("holdout manifest needs 40-128 residue examples in both val and test")
    combined = [(f"C{i}", row, False) for i, row in enumerate(candidates)] + [
        (f"H{i}", row, True) for i, row in enumerate(holdouts)]
    with tempfile.TemporaryDirectory(prefix="flowfold-cluster-") as temporary:
        root = Path(temporary)
        fasta = root / "all.fasta"
        with fasta.open("w") as stream:
            for key, row, _ in combined:
                stream.write(f">{key}\n{row['sequence']}\n")
        prefix = root / "clustered"
        subprocess.run(["mmseqs", "easy-cluster", str(fasta), str(prefix), str(root / "tmp"),
                        "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
                        "--threads", str(args.threads)], check=True)
        # Greedy clustering only compares members through representatives.
        # Search every candidate against every holdout as an extra leakage gate.
        candidate_fasta = root / "candidates.fasta"
        holdout_fasta = root / "holdouts.fasta"
        with candidate_fasta.open("w") as stream:
            for key, row, heldout in combined:
                if not heldout:
                    stream.write(f">{key}\n{row['sequence']}\n")
        with holdout_fasta.open("w") as stream:
            for key, row, heldout in combined:
                if heldout:
                    stream.write(f">{key}\n{row['sequence']}\n")
        hits = root / "holdout_hits.tsv"
        subprocess.run(["mmseqs", "easy-search", str(candidate_fasta), str(holdout_fasta),
                        str(hits), str(root / "search_tmp"),
                        "--min-seq-id", "0.3", "-c", "0.8", "--cov-mode", "0",
                        "-s", "7.5", "--format-output", "query,target",
                        "--threads", str(args.threads)], check=True)
        direct_hits = {line.split("\t", 1)[0] for line in hits.read_text().splitlines()}
        cluster = {}
        for line in prefix.with_name(prefix.name + "_cluster.tsv").read_text().splitlines():
            representative, member = line.split("\t")[:2]
            cluster[member] = representative
    if len(cluster) != len(combined):
        raise ValueError("mmseqs clustering did not assign every sequence")
    blocked = {cluster[key] for key, _, heldout in combined if heldout}
    members = defaultdict(list)
    for key, row, heldout in combined:
        if not heldout and cluster[key] not in blocked and key not in direct_hits:
            members[cluster[key]].append(row)
    rng = random.Random(args.seed)
    groups = list(members.items())
    rng.shuffle(groups)
    train = []
    for representative, rows in groups:
        rng.shuffle(rows)
        for row in rows:
            if len(train) == args.target:
                break
            train.append(dict(row, split="train", cluster=representative))
        if len(train) == args.target:
            break
    if len(train) < args.target:
        raise ValueError(f"only {len(train)} candidates remain after holdout cluster exclusion; "
                         f"need {args.target}. Collect more or lower the stated target")
    # Rebase every path to the final manifest, so moving the complete work/
    # and data/cif/ trees to another machine leaves the manifest valid.
    candidate_base = args.candidates.resolve().parent
    holdout_base = Path(__file__).resolve().parent.parent / "outputs"
    output_base = args.out.resolve().parent
    for row in train:
        source = Path(row["path"])
        source = source if source.is_absolute() else candidate_base / source
        row["path"] = os.path.relpath(source.resolve(), output_base)
    portable_holdouts = []
    for row in holdouts:
        source = Path(row["path"])
        source = source if source.is_absolute() else holdout_base / source
        portable_holdouts.append(dict(row, path=os.path.relpath(source.resolve(), output_base)))
    result = dict(source="AFDB high-confidence train; experimental PDB val/test",
                  path_base="manifest",
                  seed=args.seed, target=args.target,
                  cluster_method="mmseqs easy-cluster --min-seq-id 0.3 -c 0.8 --cov-mode 0",
                  holdout_search="mmseqs easy-search --min-seq-id 0.3 -c 0.8 --cov-mode 0 -s 7.5",
                  direct_holdout_hits=len(direct_hits),
                  counts=dict(train=len(train), val=sum(r["split"] == "val" for r in holdouts),
                              test=sum(r["split"] == "test" for r in holdouts)),
                  proteins=train + portable_holdouts)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(args.out)
    print(json.dumps(result["counts"]))


if __name__ == "__main__":
    main()
