"""Pair geometry heads and SE(3)-equivariant residue-frame refinement.

Coordinates use units of 10 Angstroms. Rotations have local frame axes in
their columns. Pair features are invariant to global rigid transformations.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from direct_frame import rotation_6d
from pair_frame import PairFrame
from se3_flow import so3_exp


DISTANCE_EDGES_A = (2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0, 24.0, 32.0)


class FrameRefinement(nn.Module):
    """Shared residual update over residue frames; no target geometry is input."""

    def __init__(self, pair_dim=64, hidden=64, steps=4):
        super().__init__()
        if steps < 0:
            raise ValueError("steps must be nonnegative")
        self.steps = steps
        self.edge = nn.Sequential(
            nn.Linear(pair_dim + 3 + 9 + 1, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.node = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 6)
        )
        nn.init.zeros_(self.node[-1].weight)
        nn.init.zeros_(self.node[-1].bias)

    def forward(self, ca, rotation, pair, valid_mask=None, segment_ids=None):
        length = len(ca)
        if valid_mask is None:
            valid_mask = torch.ones(length, dtype=torch.bool, device=ca.device)
        if segment_ids is None:
            segment_ids = torch.zeros(length, dtype=torch.long, device=ca.device)
        allowed = valid_mask[:, None] & valid_mask[None, :]
        allowed = allowed & (segment_ids[:, None] == segment_ids[None, :])
        allowed = allowed & ~torch.eye(length, dtype=torch.bool, device=ca.device)
        mask = allowed[..., None].to(ca.dtype)
        denominator = mask.sum(dim=1).clamp_min(1)
        states = []
        for _ in range(self.steps):
            relative = ca[None, :, :] - ca[:, None, :]
            local_relative = torch.einsum("ijd,idk->ijk", relative, rotation)
            relative_rotation = rotation[:, None].transpose(-1, -2) @ rotation[None]
            distance = relative.norm(dim=-1, keepdim=True)
            geometric = torch.cat((local_relative, relative_rotation.flatten(-2), distance), dim=-1)
            message = self.edge(torch.cat((pair, geometric), dim=-1)) * mask
            pooled = message.sum(dim=1) / denominator
            delta = self.node(pooled) * valid_mask[:, None].to(ca.dtype)
            local_displacement = 0.1 * torch.tanh(delta[:, :3])
            angular_displacement = 0.15 * torch.tanh(delta[:, 3:])
            displacement = torch.einsum("li,lji->lj", local_displacement, rotation)
            ca = ca + displacement
            rotation = rotation @ so3_exp(angular_displacement)
            states.append((ca, rotation))
        return ca, rotation, states


class GeometryFrame(PairFrame):
    """PairFrame with auxiliary pair-geometry predictions and optional refinement."""

    def __init__(self, hidden=128, pair_dim=64, heads=4, layers=4,
                 refine_steps=0, refine_hidden=64):
        super().__init__(hidden=hidden, pair_dim=pair_dim, heads=heads, layers=layers)
        self.refine_steps = refine_steps
        self.distogram_head = nn.Sequential(nn.LayerNorm(pair_dim),
                                           nn.Linear(pair_dim, len(DISTANCE_EDGES_A) + 1))
        self.contact_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, 1))
        self.orientation_head = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, 9))
        self.refinement = FrameRefinement(pair_dim, refine_hidden, refine_steps) if refine_steps else None

    def _encode(self, features, aa):
        if features.ndim != 2 or features.shape != (len(aa), 320):
            raise ValueError("features must have shape [L, 320]")
        length = len(aa)
        index = torch.arange(length, device=features.device, dtype=features.dtype)
        scaled = index / max(length - 1, 1)
        position = torch.stack(
            [scaled] + [fn(scaled * 2 * math.pi * k)
                        for k in (1, 2, 4, 8) for fn in (torch.sin, torch.cos)], dim=-1
        )
        single = self.esm(features) + self.aa(aa) + self.pos(position)
        signed = (index[:, None] - index[None, :]) / max(length - 1, 1)
        relative = torch.stack((signed.abs(), signed, torch.sin(math.pi * signed)), dim=-1)
        left, right = single[:, None, :], single[None, :, :]
        pair = self.pair_init(torch.cat((
            (left + right).expand(length, length, self.hidden),
            (left - right).expand(length, length, self.hidden), relative,
        ), dim=-1))
        for block in self.blocks:
            single, pair = block(single, pair, relative)
        return single, pair, signed

    def forward_with_aux(self, features, aa, valid_mask=None, segment_ids=None):
        single, pair, signed = self._encode(features, aa)
        length = len(aa)
        ca = self.ca(single)
        ca = ca - ca[:1]
        rotation = rotation_6d(self.rot(single))
        symmetric_pair = (pair + pair.transpose(0, 1)) * 0.5
        distance_input = torch.cat((
            symmetric_pair,
            (single[:, None] + single[None, :]).expand(length, length, self.hidden),
            (single[:, None] - single[None, :]).abs().expand(length, length, self.hidden),
            signed.abs()[..., None],
        ), dim=-1)
        distance = F.softplus(self.dist(distance_input)[..., 0])
        distance = (distance + distance.T) * 0.5
        distance = distance * (1 - torch.eye(length, device=distance.device, dtype=distance.dtype))
        distogram = self.distogram_head(symmetric_pair)
        contact = self.contact_head(symmetric_pair)[..., 0]
        directional = self.orientation_head(pair)
        relative_rotation = rotation_6d(directional[..., :6])
        relative_direction = F.normalize(directional[..., 6:], dim=-1, eps=1e-6)
        states = []
        if self.refinement is not None:
            ca, rotation, states = self.refinement(ca, rotation, pair, valid_mask, segment_ids)
        aux = dict(distogram_logits=distogram, contact_logits=contact,
                   relative_rotation=relative_rotation, relative_direction=relative_direction,
                   intermediate=states, pair=pair)
        return ca, rotation, distance, aux

    def forward(self, features, aa):
        ca, rotation, distance, _ = self.forward_with_aux(features, aa)
        return ca, rotation, distance
