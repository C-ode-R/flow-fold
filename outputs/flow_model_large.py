"""Sequence-conditioned SE(3) frame velocity model for 40–128 residues.

Pair blocks use absolute clipped sequence offsets. A lightweight IPA-style
decoder consumes current frames and pair features. Translation output is in
the local residue frame; rotation output is a body-frame SO(3) tangent.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from pair_frame import PairBlock


def frame_geometry(ca, rotation):
    relative = ca[None] - ca[:, None]
    local = torch.einsum("ijd,idk->ijk", relative, rotation)
    relative_rotation = rotation[:, None].transpose(-1, -2) @ rotation[None]
    return torch.cat((local, relative_rotation.flatten(-2),
                      relative.norm(dim=-1, keepdim=True)), dim=-1)


class TriangleUpdate(nn.Module):
    def __init__(self, pair_dim):
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.left = nn.Linear(pair_dim, pair_dim)
        self.right = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pair):
        normalized = self.norm(pair)
        triangle = torch.einsum("ikc,kjc->ijc", self.left(normalized),
                                self.right(normalized)) / math.sqrt(len(pair))
        return pair + self.out(F.silu(triangle))


class GeometricAttention(nn.Module):
    def __init__(self, hidden=128, pair_dim=64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.query_point = nn.Linear(hidden, 3)
        self.key_point = nn.Linear(hidden, 3)
        self.pair_bias = nn.Linear(pair_dim, 1)
        self.pair_value = nn.Linear(pair_dim, hidden)
        self.geometry = nn.Sequential(nn.Linear(13, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.out = nn.Linear(hidden, hidden)
        self.transition = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4 * hidden),
                                        nn.SiLU(), nn.Linear(4 * hidden, hidden))
        self.point_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, single, pair, ca, rotation):
        length = len(ca)
        h = self.norm(single)
        q, k, v = self.q(h), self.k(h), self.v(h)
        qpoint = ca + torch.einsum("li,lji->lj", self.query_point(h), rotation)
        kpoint = ca + torch.einsum("li,lji->lj", self.key_point(h), rotation)
        point_squared = (qpoint[:, None] - kpoint[None]).square().sum(-1)
        logits = q @ k.T / math.sqrt(q.shape[-1]) + self.pair_bias(pair)[..., 0]
        logits = logits - F.softplus(self.point_scale) * point_squared
        attention = logits.softmax(-1)
        geometry = frame_geometry(ca, rotation)
        message = v[None] + self.pair_value(pair) + self.geometry(geometry)
        update = (attention[..., None] * message).sum(1)
        single = single + self.out(update)
        return single + self.transition(single)


class SequenceFrameFlow(nn.Module):
    def __init__(self, esm_dim=320, hidden=128, pair_dim=64, heads=4,
                 pair_layers=4, geometric_layers=2):
        super().__init__()
        self.config = dict(esm_dim=esm_dim, hidden=hidden, pair_dim=pair_dim,
                           heads=heads, pair_layers=pair_layers,
                           geometric_layers=geometric_layers)
        self.esm = nn.Linear(esm_dim, hidden)
        self.aa = nn.Embedding(20, hidden)
        self.position = nn.Linear(9, hidden)
        self.time = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.pair_init = nn.Sequential(nn.Linear(2 * hidden + 3, pair_dim),
                                       nn.SiLU(), nn.LayerNorm(pair_dim))
        self.pair_blocks = nn.ModuleList(PairBlock(hidden, pair_dim, heads)
                                         for _ in range(pair_layers))
        self.geometry_pair = nn.Linear(13, pair_dim)
        self.geometry_gate = nn.Parameter(torch.tensor(0.0))
        self.triangle = TriangleUpdate(pair_dim)
        self.geometric_blocks = nn.ModuleList(GeometricAttention(hidden, pair_dim)
                                              for _ in range(geometric_layers))
        self.velocity = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden),
                                      nn.SiLU(), nn.Linear(hidden, 6))
        self.distogram = nn.Sequential(nn.LayerNorm(pair_dim), nn.Linear(pair_dim, 11))
        nn.init.zeros_(self.velocity[-1].weight)
        nn.init.zeros_(self.velocity[-1].bias)

    def encode(self, embed, aa, time):
        length = len(aa)
        index = torch.arange(length, device=embed.device, dtype=embed.dtype)
        scaled = index / max(length - 1, 1)
        position = torch.stack([scaled] + [fn(scaled * 2 * math.pi * k)
                         for k in (1, 2, 4, 8) for fn in (torch.sin, torch.cos)], -1)
        time_features = torch.stack((time, torch.sin(2 * math.pi * time),
                                     torch.cos(2 * math.pi * time)))
        single = self.esm(embed) + self.aa(aa) + self.position(position)
        single = single + self.time(time_features)[None]
        offset = index[:, None] - index[None]
        clipped = offset.clamp(-32, 32) / 32
        relative = torch.stack((clipped.abs(), clipped, torch.sign(offset)), -1)
        left = single[:, None]
        right = single[None]
        pair = self.pair_init(torch.cat(((left + right).expand(length, length, -1),
                                         (left - right).expand(length, length, -1), relative), -1))
        for block in self.pair_blocks:
            single, pair = block(single, pair, relative)
        return single, pair

    def forward(self, embed, aa, ca, rotation, time):
        single, pair = self.encode(embed, aa, time)
        pair = pair + torch.tanh(self.geometry_gate) * self.geometry_pair(frame_geometry(ca, rotation))
        pair = self.triangle(pair)
        for block in self.geometric_blocks:
            single = block(single, pair, ca, rotation)
        raw = self.velocity(single)
        global_velocity = torch.einsum("li,lji->lj", raw[:, :3], rotation)
        global_velocity = global_velocity - global_velocity.mean(0, keepdim=True)
        local_velocity = torch.einsum("lj,lji->li", global_velocity, rotation)
        symmetric = (pair + pair.transpose(0, 1)) * 0.5
        return local_velocity, raw[:, 3:], self.distogram(symmetric)
