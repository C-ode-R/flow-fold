"""Train geometry-supervised PairFrame with optional iterative refinement.

Only train and validation structures are loaded. Test is reserved for the
separate geometry_compare.py command after all checkpoint selection is final.
"""
import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from direct_frame import load_protein, losses
from geometry_frame import GeometryFrame
from geometry_metrics import evaluate_group, summarize
from geometry_targets import geometry_losses, target_geometry
from pair_train import fingerprint, seed_everything, update_best, validation_score


def intermediate_loss(states, protein, local_table):
    if not states:
        return protein["ca"].sum() * 0
    target_ca = protein["ca"]
    target_pair = torch.cdist(target_ca, target_ca)
    local = local_table[protein["aa"]]
    output = target_ca.new_zeros(())
    for ca, rotation in states:
        anchored = ca - ca[:1]
        atoms = anchored[:, None] + torch.einsum("lai,lji->laj", local, rotation)
        output = output + F.smooth_l1_loss(anchored, target_ca, beta=0.5)
        output = output + F.smooth_l1_loss(torch.cdist(ca, ca), target_pair, beta=0.5)
        output = output + 0.5 * F.smooth_l1_loss(atoms, protein["atoms"], beta=0.5)
    return output / len(states)


def read_group(rows, features, device, split):
    proteins, skipped = [], []
    for row in rows:
        if row["split"] != split:
            continue
        try:
            protein = load_protein(row, features, device)
            if not all(torch.isfinite(protein[key]).all() for key in ("ca", "rot", "atoms", "embed")):
                raise ValueError("non-finite structure or embedding")
            protein["geometry_target"] = target_geometry(protein)
            if not protein["geometry_target"]["distance_mask"].any():
                raise ValueError("no valid residue pairs")
            proteins.append(protein)
        except (ValueError, KeyError, IndexError) as error:
            skipped.append(dict(id=row.get("id"), chain=row.get("chain"), reason=str(error)))
    return proteins, skipped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--esm-cache", type=Path, required=True)
    parser.add_argument("--init-checkpoint", type=Path, required=True,
                        help="Pair-Aware A checkpoint; B and C must use the same file")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=0.001)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--refine-steps", type=int, default=0)
    parser.add_argument("--intermediate-weight", type=float, default=0.0)
    parser.add_argument("--distogram-weight", type=float, default=0.2)
    parser.add_argument("--contact-weight", type=float, default=0.1)
    parser.add_argument("--orientation-weight", type=float, default=0.05)
    parser.add_argument("--direction-weight", type=float, default=0.05)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.patience < 1 or args.refine_steps < 0:
        parser.error("steps, eval-every, and patience must be positive; refine-steps nonnegative")
    if min(args.intermediate_weight, args.distogram_weight, args.contact_weight,
           args.orientation_weight, args.direction_weight) < 0:
        parser.error("loss weights must be nonnegative")
    seed_everything(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    features = torch.load(args.esm_cache, map_location="cpu", weights_only=True)["features"]
    groups, skipped = {}, {}
    for split in ("train", "val"):
        groups[split], skipped[split] = read_group(manifest["proteins"], features, device, split)
        if not groups[split]:
            raise ValueError(f"No valid proteins in {split}")
    checkpoint = torch.load(args.init_checkpoint, map_location=device, weights_only=True)
    if checkpoint.get("architecture") != "pair_frame":
        raise ValueError("Initialization checkpoint must come from PairFrame")
    if checkpoint.get("manifest_sha256") != fingerprint(args.manifest):
        raise ValueError("Initialization checkpoint used a different manifest")
    if checkpoint.get("esm_cache_sha256") != fingerprint(args.esm_cache):
        raise ValueError("Initialization checkpoint used a different ESM cache")
    config = dict(checkpoint["config"], refine_steps=args.refine_steps)
    model = GeometryFrame(**config).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model"], strict=False)
    if unexpected or any(not key.startswith(("distogram_head.", "contact_head.",
                                             "orientation_head.", "refinement.")) for key in missing):
        raise ValueError(f"Unexpected initialization keys: missing={missing}, unexpected={unexpected}")
    local_table = checkpoint["local_table"].to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    shuffler = random.Random(args.seed)
    order = list(range(len(groups["train"])))
    best_score, best_step, stale = float("inf"), 0, 0
    history = []
    weights = dict(distogram=args.distogram_weight, contact=args.contact_weight,
                   orientation=args.orientation_weight, direction=args.direction_weight)
    metadata = dict(architecture="geometry_frame", config=config, seed=args.seed,
                    steps_cap=args.steps, lr=args.lr, geometry_weights=weights,
                    intermediate_weight=args.intermediate_weight,
                    init_checkpoint=str(args.init_checkpoint),
                    init_sha256=fingerprint(args.init_checkpoint),
                    manifest_sha256=fingerprint(args.manifest),
                    esm_cache_sha256=fingerprint(args.esm_cache),
                    split_counts={name: len(group) for name, group in groups.items()},
                    skipped=skipped, selection="median_CA_RMSD_A/10 + (1 - median_lDDT_CA)")
    for step in range(1, args.steps + 1):
        if (step - 1) % len(order) == 0:
            shuffler.shuffle(order)
        protein = groups["train"][order[(step - 1) % len(order)]]
        model.train()
        ca, rotation, distance, aux = model.forward_with_aux(protein["embed"], protein["aa"])
        anchored = ca - ca[:1]
        structure, parts = losses(lambda _features, _aa: (anchored, rotation, distance),
                                  protein, local_table)
        geometry = geometry_losses(aux, protein["geometry_target"])
        intermediate = intermediate_loss(aux["intermediate"], protein, local_table) if args.intermediate_weight else structure.new_zeros(())
        total = structure + sum(weights[key] * geometry[key] for key in weights)
        total = total + args.intermediate_weight * intermediate
        optimizer.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % args.eval_every == 0 or step == args.steps:
            val_rows = evaluate_group(model, groups["val"], local_table)
            val_summary = summarize(val_rows)
            score = validation_score(val_summary)
            record = dict(step=step, train_loss=float(total.detach()),
                          structure_loss=float(structure.detach()),
                          geometry_losses={key: float(value.detach()) for key, value in geometry.items()},
                          intermediate_loss=float(intermediate.detach()),
                          train_parts=parts, val_score=score, val_summary=val_summary)
            history.append(record)
            print(json.dumps({key: value for key, value in record.items()
                              if key not in ("train_parts", "val_summary")}), flush=True)
            best_score, stale, improved = update_best(score, best_score, stale, args.min_delta)
            if improved:
                best_step = step
                saved = dict(model=model.state_dict(), local_table=local_table.cpu(),
                             step=step, val_score=score, val_summary=val_summary, **metadata)
                temporary = args.out / "best_checkpoint.pt.tmp"
                torch.save(saved, temporary)
                temporary.replace(args.out / "best_checkpoint.pt")
            (args.out / "history.json").write_text(json.dumps(dict(
                metadata=metadata, best_step=best_step, best_score=best_score,
                stopped_at_step=step, history=history), indent=2))
            if stale >= args.patience:
                print(f"Early stopping at {step}; best {best_step}", flush=True)
                break


if __name__ == "__main__":
    main()
