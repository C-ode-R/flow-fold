"""Evaluate DirectFrame and PairFrame checkpoints under identical conditions.

Run with ``--splits val test`` only after model selection is finished. The
training entry point never loads or evaluates the test split.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from direct_frame import DirectFrame, load_protein
from frame_metrics import evaluate_group, summarize
from pair_frame import PairFrame


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(run, device):
    path = run / "best_checkpoint.pt"
    if not path.is_file():
        path = run / "checkpoint.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    architecture = checkpoint.get("architecture", "direct_frame")
    if architecture == "pair_frame":
        model = PairFrame(**checkpoint["config"])
    elif architecture == "direct_frame":
        model = DirectFrame()
    else:
        raise ValueError(f"Unknown architecture {architecture!r} in {path}")
    model.load_state_dict(checkpoint["model"])
    model = model.to(device).eval()
    return model, checkpoint["local_table"].to(device), architecture, str(path), checkpoint


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    manifest = json.loads(args.manifest.read_text())
    manifest_hash = fingerprint(args.manifest)
    cache_hash = fingerprint(args.esm_cache)
    features = torch.load(args.esm_cache, map_location="cpu", weights_only=True)["features"]
    groups = {
        split: [load_protein(row, features, device) for row in manifest["proteins"] if row["split"] == split]
        for split in dict.fromkeys(args.splits)
    }
    results = {}
    for run in args.runs:
        model, local_table, architecture, checkpoint_path, checkpoint = load_checkpoint(run, device)
        if checkpoint.get("manifest_sha256", manifest_hash) != manifest_hash:
            raise ValueError(f"Manifest differs from the one used for {run}")
        if checkpoint.get("esm_cache_sha256", cache_hash) != cache_hash:
            raise ValueError(f"ESM cache differs from the one used for {run}")
        split_results = {split: evaluate_group(model, group, local_table) for split, group in groups.items()}
        summary = {split: summarize(rows) for split, rows in split_results.items()}
        key = run.name
        if key in results:
            raise ValueError(f"Duplicate run name {key!r}")
        results[key] = dict(architecture=architecture, checkpoint=checkpoint_path,
                            summary=summary, results=split_results)
        print(key, json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(manifest=str(args.manifest), esm_cache=str(args.esm_cache),
                                        splits=list(groups), runs=results), indent=2))


if __name__ == "__main__":
    main()
