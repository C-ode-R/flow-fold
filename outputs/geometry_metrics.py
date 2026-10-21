"""Common held-out metrics for PairFrame and geometry-supervised variants."""
import numpy as np
import torch
from torch.nn import functional as F

from frame_metrics import evaluate_one as evaluate_frame_one
from frame_metrics import summarize as summarize_frame
from geometry_frame import GeometryFrame
from geometry_targets import target_geometry


OPTIONAL_HEAD_METRICS = (
    "contact_head_precision_L", "distogram_cross_entropy", "distogram_accuracy",
    "contact_cross_entropy", "relative_orientation_mae_deg", "relative_direction_cosine",
)


def fixed_correspondence_tm_score(predicted, target):
    """TM-score with fixed residue correspondence and approximate rigid optimization.

    This uses the standard length normalization and tries global plus fragment
    initial superpositions with iterative reweighting. It is not TM-align,
    which can change residue correspondence.
    """
    x = np.asarray(predicted, dtype=np.float64)
    y = np.asarray(target, dtype=np.float64)
    length = len(x)
    d0 = max(0.5, 1.24 * np.cbrt(max(length - 15, 0)) - 1.8)

    def align(weights):
        weights = weights / weights.sum()
        center_x = (x * weights[:, None]).sum(0)
        center_y = (y * weights[:, None]).sum(0)
        u, _, vt = np.linalg.svd(((x - center_x) * weights[:, None]).T @ (y - center_y))
        correction = np.eye(3)
        correction[-1, -1] = np.sign(np.linalg.det(u @ vt))
        return (x - center_x) @ (u @ correction @ vt) + center_y

    span = min(length, max(8, length // 3))
    starts = {0, max(0, (length - span) // 2), max(0, length - span)}
    initial = [np.ones(length)]
    for start in starts:
        weight = np.full(length, 1e-3)
        weight[start:start + span] = 1
        initial.append(weight)
    best = 0.0
    for weights in initial:
        for _ in range(8):
            aligned = align(weights)
            squared = ((aligned - y) ** 2).sum(-1)
            score = float((1 / (1 + squared / d0 ** 2)).mean())
            best = max(best, score)
            weights = (1 + squared / d0 ** 2) ** -2 + 1e-4
    return best


@torch.no_grad()
def evaluate_one(model, protein, local_table):
    if isinstance(model, GeometryFrame):
        ca, rotation, distance, aux = model.forward_with_aux(protein["embed"], protein["aa"])
    else:
        ca, rotation, distance = model(protein["embed"], protein["aa"])
        aux = None
    row = evaluate_frame_one(lambda _features, _aa: (ca, rotation, distance), protein, local_table)
    target = target_geometry(protein)
    length = len(ca)
    index = torch.arange(length, device=ca.device)
    upper_long = torch.triu((index[:, None] - index[None, :]).abs() >= 12, diagonal=1)
    eligible = upper_long & target["distance_mask"]
    n_top = min(length, int(eligible.sum()))
    if n_top:
        true_contact = target["distance_a"][eligible] < 8
        predicted_distance = torch.cdist(ca, ca)[eligible]
        selected = torch.topk(-predicted_distance, n_top).indices
        precision = float(true_contact[selected].float().mean())
        if aux is not None:
            head_selected = torch.topk(aux["contact_logits"][eligible], n_top).indices
            head_precision = float(true_contact[head_selected].float().mean())
        else:
            head_precision = None
        contact_density = float(true_contact.float().mean())
    else:
        precision, head_precision, contact_density = float("nan"), None, float("nan")
    if aux is not None:
        dist_mask = target["distance_mask"]
        contact_mask = target["contact_mask"]
        orientation_mask = target["orientation_mask"]
        distogram_ce = float(F.cross_entropy(aux["distogram_logits"][dist_mask],
                                             target["distogram_bins"][dist_mask])) if dist_mask.any() else None
        distogram_accuracy = float((aux["distogram_logits"][dist_mask].argmax(-1) ==
                                    target["distogram_bins"][dist_mask]).float().mean()) if dist_mask.any() else None
        contact_ce = float(F.binary_cross_entropy_with_logits(
            aux["contact_logits"][contact_mask], target["contact"][contact_mask])) if contact_mask.any() else None
        if orientation_mask.any():
            predicted_rotation = aux["relative_rotation"][orientation_mask]
            true_rotation = target["relative_rotation"][orientation_mask]
            trace = torch.einsum("nij,nij->n", predicted_rotation, true_rotation)
            orientation_mae = float(torch.rad2deg(torch.acos(((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6))).mean())
            direction_cosine = float((aux["relative_direction"][orientation_mask] *
                                      target["relative_direction"][orientation_mask]).sum(-1).mean())
        else:
            orientation_mae, direction_cosine = None, None
    else:
        distogram_ce = distogram_accuracy = contact_ce = orientation_mae = direction_cosine = None
    row.update(
        tm_score_fixed=fixed_correspondence_tm_score((ca * 10).cpu().numpy(),
                                                      (protein["ca"] * 10).cpu().numpy()),
        contact_precision_L=precision,
        contact_head_precision_L=head_precision,
        distogram_cross_entropy=distogram_ce,
        distogram_accuracy=distogram_accuracy,
        contact_cross_entropy=contact_ce,
        relative_orientation_mae_deg=orientation_mae,
        relative_direction_cosine=direction_cosine,
        native_long_contact_density=contact_density,
    )
    return row


@torch.no_grad()
def evaluate_group(model, proteins, local_table):
    was_training = model.training
    model.eval()
    rows = [evaluate_one(model, protein, local_table) for protein in proteins]
    model.train(was_training)
    return rows


def summarize(rows):
    # The common summary also includes the new metrics when they are numeric.
    result = summarize_frame([{key: value for key, value in row.items()
                               if key not in OPTIONAL_HEAD_METRICS} for row in rows])
    for key in OPTIONAL_HEAD_METRICS:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.nanmedian(values)) if values else None
    return result
