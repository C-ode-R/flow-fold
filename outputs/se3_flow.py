"""Single-protein SE(3) residue-frame flow matching experiment.

Example: python se3_flow.py --cif 1CRN.cif --out run_se3 --steps 3000
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1, seq3
from torch import nn
from torch.nn import functional as F

from flow_fold import ALPHABET, kabsch_rmsd, random_rotations


ATOM_NAMES = ("N", "CA", "C", "CB")


def load_frames(path, chain_id):
    chain = MMCIFParser(QUIET=True).get_structure("protein", str(path))[0][chain_id]
    residues = [r for r in chain if r.id[0] == " " and all(a in r for a in ("N", "CA", "C"))]
    sequence = "".join(seq1(r.resname, custom_map={"UNK": "X"}) for r in residues)
    if len(residues) < 3 or any(a not in ALPHABET for a in sequence):
        raise ValueError("Need at least three standard amino acids with N, CA and C atoms")
    ca = torch.tensor(np.stack([r["CA"].coord for r in residues]), dtype=torch.float32)
    ca = ca - ca.mean(0, keepdim=True)
    scale = torch.sqrt(ca.square().sum(-1).mean())
    ca = ca / scale
    frames, templates, cb_mask = [], [], []
    for residue in residues:
        a = torch.tensor(residue["CA"].coord, dtype=torch.float32)
        n = (torch.tensor(residue["N"].coord, dtype=torch.float32) - a) / scale
        c = (torch.tensor(residue["C"].coord, dtype=torch.float32) - a) / scale
        e1 = F.normalize(c, dim=0)
        e2 = F.normalize(n - (n * e1).sum() * e1, dim=0)
        e3 = torch.cross(e1, e2, dim=0)
        rotation = torch.stack((e1, e2, e3), dim=-1)
        frames.append(rotation)
        atoms = []
        for name in ATOM_NAMES:
            if name in residue:
                atoms.append(((torch.tensor(residue[name].coord, dtype=torch.float32) - a) / scale) @ rotation)
            else:
                atoms.append(torch.zeros(3))
        templates.append(torch.stack(atoms))
        cb_mask.append("CB" in residue)
    return sequence, ca, torch.stack(frames), torch.stack(templates), torch.tensor(cb_mask), float(scale)


def skew(w):
    x, y, z = w.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*w.shape[:-1], 3, 3)


def so3_exp(w):
    angle = w.norm(dim=-1, keepdim=True)
    small = angle < 1e-4
    a = torch.where(small, 1 - angle.square() / 6 + angle.pow(4) / 120,
                    torch.sin(angle) / angle.clamp_min(1e-8))
    b = torch.where(small, 0.5 - angle.square() / 24 + angle.pow(4) / 720,
                    (1 - torch.cos(angle)) / angle.square().clamp_min(1e-8))
    k = skew(w)
    eye = torch.eye(3, device=w.device, dtype=w.dtype)
    return eye + a[..., None] * k + b[..., None] * (k @ k)


def so3_log(rotation):
    trace = rotation.diagonal(dim1=-2, dim2=-1).sum(-1)
    cosine = ((trace - 1) / 2).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cosine)
    vee = torch.stack((rotation[..., 2, 1] - rotation[..., 1, 2],
                       rotation[..., 0, 2] - rotation[..., 2, 0],
                       rotation[..., 1, 0] - rotation[..., 0, 1]), -1)
    factor = torch.where(angle < 1e-4, 0.5 + angle.square() / 12,
                         angle / (2 * torch.sin(angle).clamp_min(1e-6)))
    return vee * factor[..., None]


def reconstruct(ca, rotation, local_atoms):
    return ca[..., None, :] + torch.einsum("...lai,...lji->...laj", local_atoms, rotation)


def generic_helix(length, device):
    """A sequence-length-only source backbone; contains no target coordinates."""
    index = torch.arange(length, device=device, dtype=torch.float32)
    angle = index * (100 * math.pi / 180)
    radius, rise = 2.3, 1.5
    xyz = torch.stack((radius * torch.cos(angle), radius * torch.sin(angle), rise * index), -1)
    xyz = xyz - xyz.mean(0, keepdim=True)
    xyz = xyz / torch.sqrt(xyz.square().sum(-1).mean())
    tangent = F.normalize(torch.stack((-radius * torch.sin(angle) * (100 * math.pi / 180),
                                       radius * torch.cos(angle) * (100 * math.pi / 180),
                                       torch.full_like(angle, rise)), -1), dim=-1)
    inward = torch.stack((-torch.cos(angle), -torch.sin(angle), torch.zeros_like(angle)), -1)
    second = F.normalize(inward - (inward * tangent).sum(-1, keepdim=True) * tangent, dim=-1)
    third = torch.cross(tangent, second, dim=-1)
    return xyz, torch.stack((tangent, second, third), -1)


def helix_source(batch, length, device, q=None):
    base_ca, base_rotation = generic_helix(length, device)
    if q is None:
        q = random_rotations(batch, device)
    canonical_ca = base_ca[None] + 0.10 * torch.randn(batch, length, 3, device=device)
    canonical_ca = canonical_ca - canonical_ca.mean(1, keepdim=True)
    source_ca = canonical_ca @ q
    perturbation = so3_exp(0.20 * torch.randn(batch, length, 3, device=device))
    source_rotation = q.transpose(-1, -2)[:, None] @ base_rotation[None] @ perturbation
    return source_ca, source_rotation, q


class FrameFlow(nn.Module):
    def __init__(self, length, hidden=64, layers=3):
        super().__init__()
        self.aa = nn.Embedding(len(ALPHABET), hidden)
        self.position = nn.Embedding(length, hidden)
        self.time = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.edges = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden + 13, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
            for _ in range(layers)
        ])
        self.nodes = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            for _ in range(layers)
        ])
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 6))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        ids = torch.arange(length)
        self.register_buffer("positions", ids)
        self.register_buffer("separation", (ids[:, None] - ids[None, :]).abs().float()[None, :, :, None] / length)
        self.register_buffer("mask", (ids[:, None] != ids[None, :]).float()[None, :, :, None])

    def forward(self, ca, rotation, time, sequence_ids):
        batch, length, _ = ca.shape
        time = time.reshape(batch, 1)
        tf = torch.cat((time, torch.sin(2 * math.pi * time), torch.cos(2 * math.pi * time)), -1)
        h = self.aa(sequence_ids)[None] + self.position(self.positions)[None] + self.time(tf)[:, None]
        h = h.expand(batch, -1, -1)
        relative_xyz = torch.einsum("bijd,bidk->bijk", ca[:, None] - ca[:, :, None], rotation)
        relative_rotation = rotation[:, :, None].transpose(-1, -2) @ rotation[:, None]
        geometric = torch.cat((relative_xyz, relative_rotation.flatten(-2),
                               self.separation.expand(batch, -1, -1, -1)), -1)
        for edge, node in zip(self.edges, self.nodes):
            hi = h[:, :, None].expand(-1, -1, length, -1)
            hj = h[:, None].expand(-1, length, -1, -1)
            messages = edge(torch.cat((hi, hj, geometric), -1)) * self.mask
            h = h + node(torch.cat((h, messages.sum(2) / (length - 1)), -1))
        output = self.head(h)
        global_v = torch.einsum("bli,blji->blj", output[..., :3], rotation)
        global_v = global_v - global_v.mean(1, keepdim=True)
        local_v = torch.einsum("blj,blji->bli", global_v, rotation)
        return local_v, output[..., 3:]


@torch.no_grad()
def generate(model, ids, length, ode_steps, seed, device, prior="gaussian_frames"):
    torch.manual_seed(seed)
    if prior == "helix":
        ca, rotation, _ = helix_source(1, length, device)
    else:
        ca = torch.randn(1, length, 3, device=device)
        ca = ca - ca.mean(1, keepdim=True)
        rotation = random_rotations(length, device).reshape(1, length, 3, 3)
    model.eval()
    dt = 1 / ode_steps
    for step in range(ode_steps):
        t = torch.tensor([step * dt], device=device)
        local_v, w = model(ca, rotation, t, ids)
        ca = ca + dt * torch.einsum("bli,blji->blj", local_v, rotation)
        rotation = rotation @ so3_exp(dt * w)
    return ca[0], rotation[0]


def write_pdb(path, sequence, atoms, cb_mask):
    with open(path, "w") as handle:
        serial = 1
        for i, aa in enumerate(sequence):
            for atom_index, name in enumerate(ATOM_NAMES):
                if name == "CB" and not cb_mask[i]:
                    continue
                x, y, z = atoms[i, atom_index].tolist()
                handle.write(f"ATOM   {serial:4d} {name:4s} {seq3(aa).upper():3s} A{i+1:4d}    "
                             f"{x:8.3f}{y:8.3f}{z:8.3f}{1.:6.2f}{0.:6.2f}           {name[0]}\n")
                serial += 1
        handle.write("END\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--out", type=Path, default=Path("run_se3"))
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--eval-seeds", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coupling", choices=("independent", "aligned"), default="aligned")
    parser.add_argument("--prior", choices=("gaussian_frames", "helix"), default="gaussian_frames")
    parser.add_argument("--atom-loss-weight", type=float, default=0.0)
    parser.add_argument("--bond-loss-weight", type=float, default=0.0)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence, ca_cpu, rotation_cpu, templates_cpu, cb_mask, scale = load_frames(args.cif, args.chain)
    ca, rotation, templates = ca_cpu.to(device), rotation_cpu.to(device), templates_cpu.to(device)
    ids = torch.tensor([ALPHABET.index(a) for a in sequence], device=device)
    model = FrameFlow(len(sequence)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    previous_steps = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["sequence"] != sequence:
            raise ValueError("Checkpoint sequence does not match input protein")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        previous_steps = checkpoint.get("steps", 0)
    history = []
    atom_mask = torch.ones(len(sequence), len(ATOM_NAMES), 1, device=device)
    atom_mask[:, 3, 0] = cb_mask.to(device)
    for step in range(previous_steps + 1, previous_steps + args.steps + 1):
        batch = args.batch
        if args.prior == "helix":
            source_ca, source_rotation, q = helix_source(batch, len(sequence), device)
        else:
            source_ca = torch.randn(batch, len(sequence), 3, device=device)
            source_ca = source_ca - source_ca.mean(1, keepdim=True)
            if args.coupling == "aligned":
                covariance = ca.T.unsqueeze(0) @ source_ca
                u, _, vh = torch.linalg.svd(covariance)
                correction = torch.eye(3, device=device).expand(batch, -1, -1).clone()
                correction[:, 2, 2] = torch.sign(torch.linalg.det(u @ vh))
                q = u @ correction @ vh
            else:
                q = random_rotations(batch, device)
            source_rotation = random_rotations(batch * len(sequence), device).reshape(batch, len(sequence), 3, 3)
        target_ca = ca[None] @ q
        target_rotation = q.transpose(-1, -2)[:, None] @ rotation[None]
        omega = so3_log(source_rotation.transpose(-1, -2) @ target_rotation)
        time = torch.rand(batch, device=device)
        frame_t = source_rotation @ so3_exp(time[:, None, None] * omega)
        ca_t = (1 - time[:, None, None]) * source_ca + time[:, None, None] * target_ca
        target_v = torch.einsum("blj,blji->bli", target_ca - source_ca, frame_t)
        predicted_v, predicted_w = model(ca_t, frame_t, time, ids)
        position_loss = F.mse_loss(predicted_v, target_v)
        rotation_loss = F.mse_loss(predicted_w, omega)
        loss = position_loss + rotation_loss
        if args.atom_loss_weight or args.bond_loss_weight:
            predicted_global_v = torch.einsum("bli,blji->blj", predicted_v, frame_t)
            estimated_ca = ca_t + (1 - time[:, None, None]) * predicted_global_v
            estimated_rotation = frame_t @ so3_exp((1 - time[:, None, None]) * predicted_w)
            estimated_atoms = reconstruct(estimated_ca, estimated_rotation, templates)
            target_atoms = reconstruct(target_ca, target_rotation, templates)
            if args.atom_loss_weight:
                atom_loss = (((estimated_atoms - target_atoms).square()) * atom_mask).sum() / (atom_mask.sum() * batch * 3)
                loss = loss + args.atom_loss_weight * atom_loss
            if args.bond_loss_weight:
                estimated_bond = (estimated_atoms[:, 1:, 0] - estimated_atoms[:, :-1, 2]).norm(dim=-1)
                target_bond = (target_atoms[:, 1:, 0] - target_atoms[:, :-1, 2]).norm(dim=-1)
                bond_loss = F.mse_loss(estimated_bond, target_bond)
                loss = loss + args.bond_loss_weight * bond_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0 or step == args.steps:
            row = {"step": step, "position_mse": float(position_loss.detach()),
                   "rotation_mse": float(rotation_loss.detach())}
            if args.atom_loss_weight:
                row["atom_mse"] = float(atom_loss.detach())
            if args.bond_loss_weight:
                row["bond_mse"] = float(bond_loss.detach())
            history.append(row)
            print(f"step {step:5d}  pos {row['position_mse']:.4f}  rot {row['rotation_mse']:.4f}", flush=True)
    mirror = ca * torch.tensor([-1., 1., 1.], device=device)
    evaluations = []
    first_atoms = None
    for seed in range(args.seed + 1, args.seed + 1 + args.eval_seeds):
        generated_ca, generated_rotation = generate(model, ids, len(sequence), args.ode_steps, seed, device, args.prior)
        atoms = reconstruct(generated_ca, generated_rotation, templates)
        if first_atoms is None:
            first_atoms = atoms
        evaluations.append({"seed": seed,
                            "target_rmsd_A": kabsch_rmsd(generated_ca, ca) * scale,
                            "mirror_rmsd_A": kabsch_rmsd(generated_ca, mirror) * scale,
                            "peptide_CN_mean_A": float((atoms[1:, 0] - atoms[:-1, 2]).norm(dim=-1).mean() * scale)})
    summary = {"sequence": sequence, "steps": previous_steps + args.steps, "device": str(device), "scale_A": scale,
               "coupling": args.coupling, "prior": args.prior, "atom_loss_weight": args.atom_loss_weight,
               "bond_loss_weight": args.bond_loss_weight,
               "median_target_rmsd_A": float(np.median([e["target_rmsd_A"] for e in evaluations])),
               "target_closer_count": sum(e["target_rmsd_A"] < e["mirror_rmsd_A"] for e in evaluations),
               "under_2A_count": sum(e["target_rmsd_A"] < 2 for e in evaluations),
               "evaluations": evaluations, "history": history}
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": previous_steps + args.steps, "sequence": sequence, "templates": templates_cpu,
                "cb_mask": cb_mask, "scale": scale, "prior": args.prior}, args.out / "checkpoint.pt")
    (args.out / "metrics.json").write_text(json.dumps(summary, indent=2))
    write_pdb(args.out / "generated_seed43.pdb", sequence, first_atoms.cpu() * scale, cb_mask)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("evaluations", "history", "sequence")}, indent=2))


if __name__ == "__main__":
    main()
