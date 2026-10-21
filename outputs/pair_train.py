"""Train PairFrame with validation-only model selection and early stopping."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch.nn import functional as F

from direct_frame import load_protein, losses, shared_local
from frame_metrics import evaluate_group, summarize
from pair_frame import PairFrame


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def fingerprint(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def consistency_loss(ca, predicted_distance):
    length = len(ca)
    if length < 2:
        return predicted_distance.sum() * 0
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ca.device), diagonal=1)
    return F.smooth_l1_loss(predicted_distance[upper], torch.cdist(ca, ca)[upper], beta=0.5)


def validation_score(summary):
    """Lower is better: 1 Å RMSD has the same weight as 0.1 lDDT."""
    return summary["ca_rmsd_A"] / 10 + (1 - summary["lddt_ca"])


def update_best(score, best_score, stale, min_delta):
    improved = score < best_score - min_delta
    return (score, 0, True) if improved else (best_score, stale + 1, False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=8, help="Validation checks without improvement")
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--pair-dim", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--consistency-weight", type=float, default=0.0)
    parser.add_argument("--bond-weight", type=float, default=0.1)
    parser.add_argument("--ca-step-weight", type=float, default=0.0)
    parser.add_argument("--rot-weight", type=float, default=0.1)
    parser.add_argument("--chiral-weight", type=float, default=0.0)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.patience < 1:
        parser.error("steps, eval-every, and patience must be positive")
    if args.consistency_weight < 0:
        parser.error("consistency-weight must be nonnegative")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    features = torch.load(args.esm_cache, map_location="cpu", weights_only=True)["features"]
    # Deliberately never load test proteins during training or model selection.
    groups = {
        split: [load_protein(row, features, device) for row in manifest["proteins"] if row["split"] == split]
        for split in ("train", "val")
    }
    if not groups["train"] or not groups["val"]:
        raise ValueError("Manifest needs nonempty train and val splits")
    local_table = shared_local(groups["train"], device)
    config = dict(hidden=args.hidden, pair_dim=args.pair_dim, heads=args.heads, layers=args.layers)
    model = PairFrame(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    ordering = list(range(len(groups["train"])))
    shuffler = random.Random(args.seed)
    best_score = float("inf")
    best_step = 0
    stale = 0
    history = []
    metadata = dict(
        architecture="pair_frame", config=config, seed=args.seed, lr=args.lr,
        consistency_weight=args.consistency_weight, bond_weight=args.bond_weight,
        ca_step_weight=args.ca_step_weight, rot_weight=args.rot_weight,
        chiral_weight=args.chiral_weight, manifest_sha256=fingerprint(args.manifest),
        esm_cache_sha256=fingerprint(args.esm_cache), split_counts={key: len(value) for key, value in groups.items()},
        selection="minimize median_CA_RMSD_A/10 + (1 - median_lDDT_CA)",
    )
    for step in range(1, args.steps + 1):
        if (step - 1) % len(ordering) == 0:
            shuffler.shuffle(ordering)
        protein = groups["train"][ordering[(step - 1) % len(ordering)]]
        model.train()
        ca, rotation, distance = model(protein["embed"], protein["aa"])
        base_loss, parts = losses(
            lambda _features, _aa: (ca, rotation, distance), protein, local_table, args.bond_weight,
            args.ca_step_weight, args.rot_weight, args.chiral_weight,
        )
        if args.consistency_weight:
            consistency = consistency_loss(ca, distance)
        else:
            consistency = base_loss.new_zeros(())
        loss = base_loss + args.consistency_weight * consistency
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            val_rows = evaluate_group(model, groups["val"], local_table)
            summary = summarize(val_rows)
            score = validation_score(summary)
            record = dict(
                step=step, train_loss=float(loss.detach()), consistency_loss=float(consistency.detach()),
                train_parts=parts, val_score=score, val_summary=summary,
            )
            history.append(record)
            print(json.dumps({key: value for key, value in record.items() if key != "train_parts"}), flush=True)
            best_score, stale, improved = update_best(score, best_score, stale, args.min_delta)
            if improved:
                best_step = step
                checkpoint = dict(model=model.state_dict(), local_table=local_table.cpu(),
                                  step=step, val_score=score, val_summary=summary, **metadata)
                temporary = args.out / "best_checkpoint.pt.tmp"
                torch.save(checkpoint, temporary)
                temporary.replace(args.out / "best_checkpoint.pt")
            (args.out / "history.json").write_text(json.dumps(dict(metadata=metadata, best_step=best_step,
                best_score=best_score, stopped_at_step=step, history=history), indent=2))
            if stale >= args.patience:
                print(f"Early stopping at step {step}; best step {best_step}", flush=True)
                break


if __name__ == "__main__":
    main()
