"""One-protein flow matching in connected backbone torsion coordinates.

Bond lengths and angles are fixed; only torsions move. Run from this directory:
python torsion_flow.py --cif 1CRN.cif --out run_torsion --steps 3000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flow_fold import ALPHABET, kabsch_rmsd, random_rotations
from se3_flow import load_frames, reconstruct, write_pdb


def wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def internal_coordinates(backbone):
    """backbone: [3L,3] ordered N, CA, C; return lengths, angles, torsions for atoms 4+."""
    a, b, c, d = backbone[:-3], backbone[1:-2], backbone[2:-1], backbone[3:]
    bc = F.normalize(c - b, dim=-1)
    n = F.normalize(torch.cross(b - a, bc, dim=-1), dim=-1)
    m = torch.cross(n, bc, dim=-1)
    cd = d - c
    lengths = cd.norm(dim=-1)
    angles = torch.acos(((-bc * cd).sum(-1) / lengths).clamp(-1 + 1e-6, 1 - 1e-6))
    radial = F.normalize(cd + (lengths * torch.cos(angles))[:, None] * bc, dim=-1)
    torsions = torch.atan2((radial * n).sum(-1), (radial * m).sum(-1))
    return lengths, angles, torsions


def extend_chain(root, lengths, angles, torsions):
    """NeRF-style serial reconstruction. root is [B,3,3], torsions [B,3L-3]."""
    points = [root[:, i] for i in range(3)]
    for index in range(torsions.shape[1]):
        a, b, c = points[-3:]
        bc = F.normalize(c - b, dim=-1)
        n = F.normalize(torch.cross(b - a, bc, dim=-1), dim=-1)
        m = torch.cross(n, bc, dim=-1)
        angle = angles[index]
        phi = torsions[:, index, None]
        direction = (-math.cos(float(angle)) * bc +
                     math.sin(float(angle)) * (torch.cos(phi) * m + torch.sin(phi) * n))
        points.append(c + lengths[index] * direction)
    return torch.stack(points, dim=1)


def source_torsions(count, device, batch, noise=0.20):
    # Added atom roles: N_i gets psi, CA_i gets omega, C_i gets phi.
    centers = torch.tensor([-45., 180., -60.], device=device) * math.pi / 180
    roles = torch.arange(3, count + 3, device=device) % 3
    return wrap(centers[roles][None] + noise * torch.randn(batch, count, device=device))


class TorsionVelocity(nn.Module):
    def __init__(self, sequence, hidden=128):
        super().__init__()
        count = 3 * len(sequence) - 3
        residue = torch.arange(3, 3 * len(sequence)) // 3
        roles = torch.arange(3, 3 * len(sequence)) % 3
        self.register_buffer("residue", residue)
        self.register_buffer("roles", roles)
        self.register_buffer("aa_ids", torch.tensor([ALPHABET.index(aa) for aa in sequence]))
        self.aa = nn.Embedding(len(ALPHABET), hidden)
        self.pos = nn.Embedding(len(sequence), hidden)
        self.role = nn.Embedding(3, hidden)
        self.input = nn.Sequential(nn.Linear(hidden + 5, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.blocks = nn.ModuleList([
            nn.Sequential(nn.Conv1d(hidden, hidden, 5, padding=2), nn.SiLU(),
                          nn.Conv1d(hidden, hidden, 1)) for _ in range(3)
        ])
        self.output = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        nn.init.zeros_(self.output[-1].weight)
        nn.init.zeros_(self.output[-1].bias)
        assert count == len(self.residue)

    def forward(self, theta, time):
        batch = theta.shape[0]
        time = time[:, None].expand_as(theta)
        features = self.aa(self.aa_ids[self.residue]) + self.pos(self.residue) + self.role(self.roles)
        dynamic = torch.stack((torch.sin(theta), torch.cos(theta), time,
                               torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)), -1)
        h = self.input(torch.cat((features[None].expand(batch, -1, -1), dynamic), -1))
        h = h.transpose(1, 2)
        for block in self.blocks:
            h = h + block(h)
        return self.output(h.transpose(1, 2)).squeeze(-1)


@torch.no_grad()
def sample(model, root, lengths, angles, steps, seed, device):
    torch.manual_seed(seed)
    theta = source_torsions(len(lengths), device, 1)
    dt = 1 / steps
    for step in range(steps):
        t = torch.tensor([step * dt], device=device)
        v1 = model(theta, t)
        v2 = model(theta + dt * v1, t + dt)
        theta = theta + 0.5 * dt * (v1 + v2)
    rotation = random_rotations(1, device)[0]
    xyz = extend_chain(root[None] @ rotation, lengths, angles, theta)[0]
    return xyz.reshape(-1, 3, 3), theta[0]


def add_cb(backbone, target_template, cb_mask):
    atoms = []
    for index in range(backbone.shape[0]):
        n, ca, c = backbone[index]
        e1 = F.normalize(c - ca, dim=0)
        n_direction = n - ca
        e2 = F.normalize(n_direction - (n_direction * e1).sum() * e1, dim=0)
        e3 = torch.cross(e1, e2, dim=0)
        frame = torch.stack((e1, e2, e3), dim=-1)
        cb = ca + target_template[index, 3] @ frame.T
        atoms.append(torch.stack((n, ca, c, cb)))
    return torch.stack(atoms)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--out", type=Path, default=Path("run_torsion"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence, ca_cpu, rotations_cpu, templates_cpu, cb_mask, scale = load_frames(args.cif, args.chain)
    ca, rotations, templates = ca_cpu.to(device), rotations_cpu.to(device), templates_cpu.to(device)
    target_atoms = reconstruct(ca, rotations, templates)
    target_backbone = target_atoms[:, :3].reshape(-1, 3)
    lengths, angles, target_torsions = internal_coordinates(target_backbone)
    root = target_backbone[:3]
    model = TorsionVelocity(sequence).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    history = []
    for step in range(1, args.steps + 1):
        theta0 = source_torsions(len(lengths), device, args.batch)
        velocity = wrap(target_torsions[None] - theta0)
        time = torch.rand(args.batch, device=device)
        theta_t = theta0 + time[:, None] * velocity
        prediction = model(theta_t, time)
        loss = F.mse_loss(prediction, velocity)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            history.append({"step": step, "velocity_mse": float(loss.detach())})
            print(f"step {step:5d}  velocity_mse {float(loss.detach()):.6f}", flush=True)
    evaluations = []
    first_atoms = None
    mirror = ca * torch.tensor([-1., 1., 1.], device=device)
    for seed in range(args.seed + 1, args.seed + 1 + args.eval_seeds):
        backbone, theta = sample(model, root, lengths, angles, args.ode_steps, seed, device)
        atoms = add_cb(backbone, templates, cb_mask)
        if first_atoms is None:
            first_atoms = atoms
        ca_generated = atoms[:, 1] - atoms[:, 1].mean(0, keepdim=True)
        bond = (atoms[1:, 0] - atoms[:-1, 2]).norm(dim=-1) * scale
        evaluations.append({"seed": seed, "target_rmsd_A": kabsch_rmsd(ca_generated, ca) * scale,
                            "mirror_rmsd_A": kabsch_rmsd(ca_generated, mirror) * scale,
                            "peptide_CN_mean_A": float(bond.mean()),
                            "peptide_CN_max_abs_error_A": float((bond - lengths[::3][:len(bond)] * scale).abs().max()),
                            "torsion_mae_rad": float(wrap(theta - target_torsions).abs().mean())})
    summary = {"sequence": sequence, "steps": args.steps, "device": str(device),
               "source": "generic alpha-helix torsions with Gaussian perturbation",
               "bond_lengths_and_angles": "fixed from 1CRN target; one-protein overfit only",
               "median_target_rmsd_A": float(np.median([e["target_rmsd_A"] for e in evaluations])),
               "target_closer_count": sum(e["target_rmsd_A"] < e["mirror_rmsd_A"] for e in evaluations),
               "under_2A_count": sum(e["target_rmsd_A"] < 2 for e in evaluations),
               "evaluations": evaluations, "history": history}
    torch.save({"model": model.state_dict(), "sequence": sequence, "lengths": lengths.cpu(),
                "angles": angles.cpu(), "root": root.cpu(), "templates": templates_cpu,
                "cb_mask": cb_mask, "scale": scale}, args.out / "checkpoint.pt")
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_pdb(args.out / "generated_seed43.pdb", sequence, first_atoms.cpu() * scale, cb_mask)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("evaluations", "history", "sequence")}, indent=2))


if __name__ == "__main__":
    main()
