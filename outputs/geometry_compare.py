"""Final common evaluation of A, B, C after validation checkpoint selection."""
import argparse
import json
from pathlib import Path

import torch

from direct_frame import load_protein
from frame_compare import fingerprint
from geometry_frame import GeometryFrame
from geometry_metrics import evaluate_group, summarize
from pair_frame import PairFrame


def load_model(run, device, manifest_hash, cache_hash):
    path = run / "best_checkpoint.pt"
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("manifest_sha256") != manifest_hash:
        raise ValueError(f"Manifest mismatch for {run}")
    if checkpoint.get("esm_cache_sha256") != cache_hash:
        raise ValueError(f"ESM cache mismatch for {run}")
    architecture = checkpoint["architecture"]
    if architecture == "pair_frame":
        model = PairFrame(**checkpoint["config"])
    elif architecture == "geometry_frame":
        model = GeometryFrame(**checkpoint["config"])
    else:
        raise ValueError(f"Unsupported architecture {architecture!r}")
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint["local_table"].to(device), checkpoint, path


def structure_groups(rows):
    bins = {"length_25_49": [], "length_50_74": [], "length_75_100": [],
            "contact_sparse": [], "contact_intermediate": [], "contact_dense": []}
    for row in rows:
        length = row["residues"]
        name = "length_25_49" if length < 50 else "length_50_74" if length < 75 else "length_75_100"
        bins[name].append(row)
        density = row["native_long_contact_density"]
        name = "contact_sparse" if density < 0.05 else "contact_intermediate" if density < 0.15 else "contact_dense"
        bins[name].append(row)
    return {name: summarize(group) for name, group in bins.items() if group}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--runs", type=Path, nargs=3, required=True,
                        help="A baseline, B geometry, C geometry + refinement")
    parser.add_argument("--control", type=Path,
                        help="Optional equal-step A continuation without geometry losses")
    parser.add_argument("--splits", choices=("val", "test"), nargs="+", default=["val"])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    manifest_hash, cache_hash = fingerprint(args.manifest), fingerprint(args.esm_cache)
    rows = json.loads(args.manifest.read_text())["proteins"]
    features = torch.load(args.esm_cache, map_location="cpu", weights_only=True)["features"]
    groups = {split: [load_protein(row, features, device) for row in rows if row["split"] == split]
              for split in dict.fromkeys(args.splits)}
    output = {}
    labelled_runs = list(zip("ABC", args.runs))
    if args.control is not None:
        labelled_runs.append(("control", args.control))
    for label, run in labelled_runs:
        model, local_table, checkpoint, path = load_model(run, device, manifest_hash, cache_hash)
        results = {split: evaluate_group(model, proteins, local_table)
                   for split, proteins in groups.items()}
        summary = {split: summarize(data) for split, data in results.items()}
        stratified = {split: structure_groups(data) for split, data in results.items()}
        output[label] = dict(run=str(run), architecture=checkpoint["architecture"],
                             checkpoint=str(path), selected_step=checkpoint["step"],
                             summary=summary, stratified=stratified, results=results)
        print(label, run.name, json.dumps(summary), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(manifest=str(args.manifest), splits=list(groups),
                                        runs=output), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
