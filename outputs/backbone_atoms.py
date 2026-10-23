"""Differentiable N/CA/C/O reconstruction from C-alpha residue frames.

Coordinates use the Flow model's 10 Å units. The frame basis is defined by
CA->C (first axis) and the component of CA->N orthogonal to it (second axis).
O uses the neighboring peptide N; terminal O is an approximate fallback.
"""
import math

import torch
from torch.nn import functional as F


SCALE = 10.0
N_CA_A = 1.458
CA_C_A = 1.525
C_O_A = 1.231
N_CA_C_DEG = 111.2


def reconstruct_backbone(ca, rotation):
    unbatched = ca.ndim == 2
    if unbatched:
        ca, rotation = ca[None], rotation[None]
    dtype, device = ca.dtype, ca.device
    angle = math.radians(N_CA_C_DEG)
    n_local = torch.tensor([N_CA_A * math.cos(angle), N_CA_A * math.sin(angle), 0.0],
                           device=device, dtype=dtype) / SCALE
    c_local = torch.tensor([CA_C_A, 0.0, 0.0], device=device, dtype=dtype) / SCALE
    n = ca + torch.einsum("bli,blji->blj", n_local.expand_as(ca), rotation)
    c = ca + torch.einsum("bli,blji->blj", c_local.expand_as(ca), rotation)
    next_n = torch.cat((n[:, 1:], n[:, -1:]), dim=1)
    toward_ca = F.normalize(ca - c, dim=-1)
    toward_next_n = F.normalize(next_n - c, dim=-1)
    fallback = F.normalize(n - c, dim=-1)
    toward_next_n = torch.cat((toward_next_n[:, :-1], fallback[:, -1:]), dim=1)
    oxygen_direction = -F.normalize(toward_ca + toward_next_n, dim=-1)
    oxygen = c + (C_O_A / SCALE) * oxygen_direction
    atoms = torch.stack((n, ca, c, oxygen), dim=-2)
    return atoms[0] if unbatched else atoms


def peptide_geometry_loss(predicted_atoms, target_atoms, segments, mask):
    valid = (mask[:, :-1] & mask[:, 1:] &
             (segments[:, :-1] == segments[:, 1:]))
    if not valid.any():
        zero = predicted_atoms.sum() * 0
        return zero, zero
    c, n_next = predicted_atoms[:, :-1, 2], predicted_atoms[:, 1:, 0]
    target_c, target_n_next = target_atoms[:, :-1, 2], target_atoms[:, 1:, 0]
    bond = ((c - n_next).norm(dim=-1) -
            (target_c - target_n_next).norm(dim=-1)).abs()[valid].mean()
    def cosine(a, b):
        return (F.normalize(a, dim=-1) * F.normalize(b, dim=-1)).sum(-1)
    predicted_angle = cosine(predicted_atoms[:, :-1, 1] - c, n_next - c)
    target_angle = cosine(target_atoms[:, :-1, 1] - target_c,
                          target_n_next - target_c)
    angle = (predicted_angle - target_angle).abs()[valid].mean()
    return bond, angle
