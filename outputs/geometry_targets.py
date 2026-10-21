"""Masked, target-derived pair geometry for the geometry-supervision runs."""
import torch
from torch.nn import functional as F

from geometry_frame import DISTANCE_EDGES_A


def target_geometry(protein, valid_residue=None, chain_break_after=None,
                    contact_threshold_a=8.0, contact_min_separation=6,
                    orientation_max_distance_a=20.0):
    """Build labels without passing target-derived masks to model inference.

    `chain_break_after[i]` means residues i and i+1 are disconnected. If it is
    absent, breaks are inferred from target C--N and adjacent CA distances for
    loss masking only. Production inference must use sequence metadata instead.
    """
    ca, rotation, atoms = protein["ca"], protein["rot"], protein["atoms"]
    length = len(ca)
    device = ca.device
    finite = torch.isfinite(ca).all(-1) & torch.isfinite(rotation).all((-1, -2))
    finite = finite & torch.isfinite(atoms).all((-1, -2))
    valid = finite if valid_residue is None else finite & valid_residue.bool()
    safe_ca = torch.where(valid[:, None], ca, torch.zeros_like(ca))
    safe_rot = torch.where(valid[:, None, None], rotation,
                           torch.eye(3, device=device, dtype=rotation.dtype))
    if chain_break_after is None:
        if length > 1:
            cn = (atoms[:-1, 2] - atoms[1:, 0]).norm(dim=-1)
            adjacent = (safe_ca[1:] - safe_ca[:-1]).norm(dim=-1)
            chain_break_after = (cn > 0.22) | (adjacent > 0.5)
        else:
            chain_break_after = torch.empty(0, dtype=torch.bool, device=device)
    if len(chain_break_after) != max(length - 1, 0):
        raise ValueError("chain_break_after must have length L-1")
    if length > 1:
        chain_break_after = chain_break_after | ~valid[:-1] | ~valid[1:]
    segment = torch.cat((torch.zeros(1, device=device, dtype=torch.long),
                         chain_break_after.long().cumsum(0)))
    index = torch.arange(length, device=device)
    separation = (index[:, None] - index[None, :]).abs()
    same_segment = segment[:, None] == segment[None, :]
    pair_valid = valid[:, None] & valid[None, :] & same_segment
    distance_a = torch.cdist(safe_ca, safe_ca) * 10
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)
    distance_mask = pair_valid & upper
    contact_mask = distance_mask & (separation >= contact_min_separation)
    orientation_mask = pair_valid & (separation >= 3) & (distance_a < orientation_max_distance_a)
    relative_rotation = safe_rot[:, None].transpose(-1, -2) @ safe_rot[None]
    displacement = safe_ca[None, :, :] - safe_ca[:, None, :]
    relative_direction = F.normalize(
        torch.einsum("ijd,idk->ijk", displacement, safe_rot), dim=-1, eps=1e-6
    )
    bins = torch.bucketize(distance_a, torch.tensor(DISTANCE_EDGES_A, device=device, dtype=distance_a.dtype))
    return dict(valid_residue=valid, segment_ids=segment, distance_a=distance_a,
                distogram_bins=bins, contact=(distance_a < contact_threshold_a).float(),
                relative_rotation=relative_rotation, relative_direction=relative_direction,
                distance_mask=distance_mask, contact_mask=contact_mask,
                orientation_mask=orientation_mask, chain_break_after=chain_break_after)


def geometry_losses(aux, target):
    logits = aux["distogram_logits"]
    mask = target["distance_mask"]
    zero = logits.sum() * 0
    distogram = F.cross_entropy(logits[mask], target["distogram_bins"][mask]) if mask.any() else zero
    mask = target["contact_mask"]
    labels = target["contact"][mask]
    if mask.any():
        per_pair = F.binary_cross_entropy_with_logits(aux["contact_logits"][mask], labels, reduction="none")
        positive = labels > 0.5
        contact = 0.5 * per_pair[positive].mean() + 0.5 * per_pair[~positive].mean() if positive.any() and (~positive).any() else per_pair.mean()
    else:
        contact = zero
    mask = target["orientation_mask"]
    if mask.any():
        predicted = aux["relative_rotation"][mask]
        true = target["relative_rotation"][mask]
        rotation = (3 - torch.einsum("nij,nij->n", predicted, true)).mean() / 3
        direction = (1 - (aux["relative_direction"][mask] * target["relative_direction"][mask]).sum(-1)).mean()
    else:
        rotation, direction = zero, zero
    return dict(distogram=distogram, contact=contact, orientation=rotation, direction=direction)
