"""Common structure metrics for DirectFrame and PairFrame checkpoints."""
import numpy as np
import torch

from flow_fold import signed_local_volume
from project_frames import measures


@torch.no_grad()
def evaluate_one(model, protein, local_table):
    ca, rotation, predicted_distance = model(protein["embed"], protein["aa"])
    result = measures(ca, rotation, protein, local_table[protein["aa"]])
    predicted_chirality = signed_local_volume(ca[None])[0, :, 0]
    target_chirality = signed_local_volume(protein["ca"][None])[0, :, 0]
    valid = target_chirality.abs() > 0.02
    result.update(
        id=protein["id"],
        chain=protein["chain"],
        residues=len(protein["seq"]),
        predicted_distance_mae_A=float(
            (predicted_distance - torch.cdist(protein["ca"], protein["ca"])).abs().mean() * 10
        ),
        distance_consistency_mae_A=float(
            (predicted_distance - torch.cdist(ca, ca)).abs().mean() * 10
        ),
        chirality_mae=float((predicted_chirality - target_chirality).abs().mean()),
        chirality_sign_accuracy=float(
            (torch.sign(predicted_chirality[valid]) == torch.sign(target_chirality[valid]))
            .float()
            .mean()
        ) if valid.any() else float("nan"),
        native_closer_than_mirror=bool(result["ca_rmsd_A"] < result["mirror_ca_rmsd_A"]),
    )
    return result


@torch.no_grad()
def evaluate_group(model, proteins, local_table):
    was_training = model.training
    model.eval()
    rows = [evaluate_one(model, protein, local_table) for protein in proteins]
    model.train(was_training)
    return rows


def summarize(rows):
    if not rows:
        raise ValueError("Cannot summarize an empty split")
    numeric = [key for key in rows[0] if key not in ("id", "chain", "residues", "native_closer_than_mirror")]
    result = {key: float(np.nanmedian([row[key] for row in rows])) for key in numeric}
    result.update(
        n=len(rows),
        native_closer_than_mirror_count=sum(row["native_closer_than_mirror"] for row in rows),
    )
    return result
