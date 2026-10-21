"""Batched SE(3) residue-frame flow model; padding is masked in every reduction."""
import math

import torch
from torch import nn
from torch.nn import functional as F


def frame_geometry(ca, rotation):
    relative = ca[:, None] - ca[:, :, None]
    local = torch.einsum("bijd,bidk->bijk", relative, rotation)
    relative_rotation = rotation[:, :, None].transpose(-1, -2) @ rotation[:, None]
    return torch.cat((local, relative_rotation.flatten(-2),
                      relative.norm(dim=-1, keepdim=True)), -1)


class BatchedPairBlock(nn.Module):
    def __init__(self, hidden, pair_dim, heads):
        super().__init__()
        self.heads = heads
        self.single_norm = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden)
        self.bias = nn.Linear(pair_dim, heads, bias=False)
        self.attention_out = nn.Linear(hidden, hidden)
        self.single_ff = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, 4 * hidden),
                                       nn.SiLU(), nn.Linear(4 * hidden, hidden))
        self.pair_norm = nn.LayerNorm(pair_dim)
        self.pair_update = nn.Sequential(nn.Linear(pair_dim + 2 * hidden + 3, pair_dim),
                                         nn.SiLU(), nn.Linear(pair_dim, pair_dim))
        nn.init.zeros_(self.pair_update[-1].weight)
        nn.init.zeros_(self.pair_update[-1].bias)

    def forward(self, single, pair, relative, mask):
        batch, length, hidden = single.shape
        q, k, v = self.qkv(self.single_norm(single)).chunk(3, -1)
        shape = (batch, length, self.heads, hidden // self.heads)
        q, k, v = (tensor.reshape(shape).transpose(1, 2) for tensor in (q, k, v))
        bias = self.bias(self.pair_norm(pair)).permute(0, 3, 1, 2)
        bias = bias.masked_fill(~mask[:, None, None, :], -1e4)
        update = F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
        update = update.transpose(1, 2).reshape(batch, length, hidden)
        single = (single + self.attention_out(update)) * mask[..., None]
        single = (single + self.single_ff(single)) * mask[..., None]
        left, right = single[:, :, None], single[:, None]
        pair_input = torch.cat((self.pair_norm(pair),
                                (left + right).expand(-1, -1, length, -1),
                                (left - right).expand(-1, -1, length, -1),
                                relative.expand(batch, -1, -1, -1)), -1)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair = (pair + self.pair_update(pair_input)) * pair_mask[..., None]
        return single, pair


class TriangleUpdate(nn.Module):
    def __init__(self, pair_dim):
        super().__init__()
        self.norm = nn.LayerNorm(pair_dim)
        self.left = nn.Linear(pair_dim, pair_dim)
        self.right = nn.Linear(pair_dim, pair_dim)
        self.out = nn.Linear(pair_dim, pair_dim)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, pair, mask):
        pair_mask = mask[:, :, None] & mask[:, None, :]
        normalized = self.norm(pair)
        left = self.left(normalized) * pair_mask[..., None]
        right = self.right(normalized) * pair_mask[..., None]
        triangle = torch.einsum("bikc,bkjc->bijc", left, right)
        triangle = triangle / mask.sum(1).clamp_min(1).sqrt()[:, None, None, None]
        return (pair + self.out(F.silu(triangle))) * pair_mask[..., None]


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

    def forward(self, single, pair, ca, rotation, mask):
        h = self.norm(single)
        q, k, v = self.q(h), self.k(h), self.v(h)
        qpoint = ca + torch.einsum("bli,blji->blj", self.query_point(h), rotation)
        kpoint = ca + torch.einsum("bli,blji->blj", self.key_point(h), rotation)
        point_squared = (qpoint[:, :, None] - kpoint[:, None]).square().sum(-1)
        logits = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(q.shape[-1])
        logits = logits + self.pair_bias(pair)[..., 0]
        logits = logits - F.softplus(self.point_scale) * point_squared
        attention = logits.masked_fill(~mask[:, None, :], -1e4).softmax(-1)
        message = v[:, None] + self.pair_value(pair) + self.geometry(frame_geometry(ca, rotation))
        update = (attention[..., None] * message).sum(2)
        single = (single + self.out(update)) * mask[..., None]
        return (single + self.transition(single)) * mask[..., None]


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
        self.pair_blocks = nn.ModuleList(BatchedPairBlock(hidden, pair_dim, heads)
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

    def encode(self, embed, aa, time, mask):
        batch, length = aa.shape
        index = torch.arange(length, device=embed.device, dtype=embed.dtype)
        lengths = mask.sum(1).clamp_min(1).to(embed.dtype)
        scaled = index[None] / (lengths[:, None] - 1).clamp_min(1)
        position = torch.stack([scaled] + [fn(scaled * 2 * math.pi * k)
                         for k in (1, 2, 4, 8) for fn in (torch.sin, torch.cos)], -1)
        time_features = torch.stack((time, torch.sin(2 * math.pi * time),
                                     torch.cos(2 * math.pi * time)), -1)
        single = self.esm(embed) + self.aa(aa) + self.position(position)
        single = (single + self.time(time_features)[:, None]) * mask[..., None]
        offset = index[:, None] - index[None]
        clipped = offset.clamp(-32, 32) / 32
        relative = torch.stack((clipped.abs(), clipped, torch.sign(offset)), -1)[None]
        left, right = single[:, :, None], single[:, None]
        pair = self.pair_init(torch.cat(((left + right).expand(-1, -1, length, -1),
                                         (left - right).expand(-1, -1, length, -1),
                                         relative.expand(batch, -1, -1, -1)), -1))
        pair_mask = mask[:, :, None] & mask[:, None, :]
        pair = pair * pair_mask[..., None]
        for block in self.pair_blocks:
            single, pair = block(single, pair, relative, mask)
        return single, pair

    def forward(self, embed, aa, ca, rotation, time, mask=None):
        unbatched = embed.ndim == 2
        if unbatched:
            embed, aa, ca, rotation = (tensor[None] for tensor in (embed, aa, ca, rotation))
            time = time.reshape(1)
        batch, length = aa.shape
        if mask is None:
            mask = torch.ones(batch, length, dtype=torch.bool, device=aa.device)
        elif unbatched:
            mask = mask[None]
        time = time.reshape(batch)
        pair_mask = mask[:, :, None] & mask[:, None, :]
        single, pair = self.encode(embed, aa, time, mask)
        pair = (pair + torch.tanh(self.geometry_gate) *
                self.geometry_pair(frame_geometry(ca, rotation))) * pair_mask[..., None]
        pair = self.triangle(pair, mask)
        for block in self.geometric_blocks:
            single = block(single, pair, ca, rotation, mask)
        raw = self.velocity(single) * mask[..., None]
        global_velocity = torch.einsum("bli,blji->blj", raw[..., :3], rotation)
        mean = global_velocity.sum(1, keepdim=True) / mask.sum(1)[:, None, None].clamp_min(1)
        global_velocity = (global_velocity - mean) * mask[..., None]
        local_velocity = torch.einsum("blj,blji->bli", global_velocity, rotation)
        symmetric = (pair + pair.transpose(1, 2)) * 0.5
        distogram = self.distogram(symmetric) * pair_mask[..., None]
        if unbatched:
            return local_velocity[0], raw[0, :, 3:], distogram[0]
        return local_velocity, raw[..., 3:], distogram
