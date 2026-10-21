"""Pair-aware direct residue-frame predictor.

Inputs and outputs use the same conventions as ``direct_frame.DirectFrame``:
ESM-2 [L, 320], residue IDs [L], and (CA [L, 3], SO(3) [L, 3, 3],
distance [L, L]) in units of 10 Angstroms.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from direct_frame import rotation_6d
from flow_fold import ALPHABET


class PairBlock(nn.Module):
    def __init__(self, hidden=128, pair_dim=64, heads=4):
        super().__init__()
        self.single_norm = nn.LayerNorm(hidden)
        self.attention = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.bias = nn.Linear(pair_dim, heads, bias=False)
        self.single_ff_norm = nn.LayerNorm(hidden)
        self.single_ff = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.SiLU(), nn.Linear(4 * hidden, hidden)
        )
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_update = nn.Sequential(
            nn.Linear(pair_dim + 2 * hidden + 3, pair_dim),
            nn.SiLU(),
            nn.Linear(pair_dim, pair_dim),
        )
        # Each block begins as an identity pair update; gradients still flow
        # through the final layer on the first optimizer step.
        nn.init.zeros_(self.pair_update[-1].weight)
        nn.init.zeros_(self.pair_update[-1].bias)

    def forward(self, single, pair, relative):
        bias = self.bias(self.pair_norm(pair)).permute(2, 0, 1)
        normalized = self.single_norm(single)[None]
        update = self.attention(
            normalized, normalized, normalized, attn_mask=bias, need_weights=False
        )[0][0]
        single = single + update
        single = single + self.single_ff(self.single_ff_norm(single))
        left = single[:, None, :]
        right = single[None, :, :]
        pair_input = torch.cat(
            (
                self.pair_norm(pair),
                (left + right).expand(pair.shape[0], pair.shape[1], single.shape[-1]),
                (left - right).expand(pair.shape[0], pair.shape[1], single.shape[-1]),
                relative,
            ),
            dim=-1,
        )
        pair = pair + self.pair_update(pair_input)
        return single, pair


class PairFrame(nn.Module):
    def __init__(self, hidden=128, pair_dim=64, heads=4, layers=4):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.hidden = hidden
        self.pair_dim = pair_dim
        self.heads = heads
        self.layers = layers
        self.esm = nn.Linear(320, hidden)
        self.aa = nn.Embedding(len(ALPHABET), hidden)
        self.pos = nn.Linear(9, hidden)
        self.pair_init = nn.Sequential(
            nn.Linear(2 * hidden + 3, pair_dim), nn.SiLU(), nn.LayerNorm(pair_dim)
        )
        self.blocks = nn.ModuleList(
            PairBlock(hidden, pair_dim, heads) for _ in range(layers)
        )
        self.ca = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 3))
        self.rot = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 6))
        self.dist = nn.Sequential(
            nn.LayerNorm(pair_dim + 2 * hidden + 1),
            nn.Linear(pair_dim + 2 * hidden + 1, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.ca[-1].weight)
        nn.init.zeros_(self.ca[-1].bias)
        nn.init.zeros_(self.rot[-1].weight)
        with torch.no_grad():
            self.rot[-1].bias.copy_(torch.tensor([1.0, 0, 0, 0, 1.0, 0]))

    def forward(self, features, aa):
        if features.ndim != 2 or features.shape != (len(aa), 320):
            raise ValueError("features must have shape [L, 320]")
        length = len(aa)
        index = torch.arange(length, device=features.device, dtype=features.dtype)
        scaled = index / max(length - 1, 1)
        position = torch.stack(
            [scaled]
            + [fn(scaled * (2 * math.pi * k)) for k in (1, 2, 4, 8) for fn in (torch.sin, torch.cos)],
            dim=-1,
        )
        single = self.esm(features) + self.aa(aa) + self.pos(position)
        signed = (index[:, None] - index[None, :]) / max(length - 1, 1)
        relative = torch.stack((signed.abs(), signed, torch.sin(math.pi * signed)), dim=-1)
        left = single[:, None, :]
        right = single[None, :, :]
        pair = self.pair_init(
            torch.cat(
                (
                    (left + right).expand(length, length, self.hidden),
                    (left - right).expand(length, length, self.hidden),
                    relative,
                ),
                dim=-1,
            )
        )
        for block in self.blocks:
            single, pair = block(single, pair, relative)
        ca = self.ca(single)
        ca = ca - ca[:1]
        rot = rotation_6d(self.rot(single))
        # Symmetric inputs and an explicit final symmetrization enforce
        # d(i,j) == d(j,i), including after floating-point rounding.
        symmetric_pair = (pair + pair.transpose(0, 1)) * 0.5
        distance_input = torch.cat(
            (
                symmetric_pair,
                (single[:, None, :] + single[None, :, :]).expand(length, length, self.hidden),
                (single[:, None, :] - single[None, :, :]).abs().expand(length, length, self.hidden),
                signed.abs()[..., None],
            ),
            dim=-1,
        )
        distance = F.softplus(self.dist(distance_input)[..., 0])
        distance = (distance + distance.T) * 0.5
        distance = distance * (1 - torch.eye(length, device=distance.device, dtype=distance.dtype))
        return ca, rot, distance
