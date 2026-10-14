"""One-protein C-alpha flow matching experiment with an equivariant graph network.

Run: python flow_fold.py --cif 1CRN.cif --steps 1200
Outputs: checkpoint.pt, metrics.json, generated_ca.pdb in --out.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from Bio.PDB import MMCIFParser
from Bio.SeqUtils import seq1
from torch import nn
from torch.nn import functional as F


ALPHABET = "ACDEFGHIKLMNPQRSTVWYX"


def load_protein(path, chain_id):
    structure = MMCIFParser(QUIET=True).get_structure("protein", str(path))
    chain = structure[0][chain_id]
    residues = [r for r in chain if r.id[0] == " " and "CA" in r]
    sequence = "".join(seq1(r.resname, custom_map={"UNK": "X"}) for r in residues)
    if len(residues) < 3 or any(a not in ALPHABET for a in sequence):
        raise ValueError("Need at least three standard residues with C-alpha coordinates")
    xyz = torch.tensor(np.stack([r["CA"].coord for r in residues]), dtype=torch.float32)
    xyz = xyz - xyz.mean(0, keepdim=True)
    scale = torch.sqrt(xyz.square().sum(-1).mean())
    return sequence, xyz / scale, float(scale)


def random_rotations(batch, device):
    q, r = torch.linalg.qr(torch.randn(batch, 3, 3, device=device))
    q = q * torch.sign(torch.diagonal(r, dim1=-2, dim2=-1)).unsqueeze(-2)
    q[:, :, 2] *= torch.where(torch.linalg.det(q) < 0, -1.0, 1.0)[:, None]
    return q


def align_target_to_noise(target, noise):
    """Rotate one fixed chiral target toward each noise sample; never reflect it."""
    batch = noise.shape[0]
    covariance = target.T.unsqueeze(0) @ noise
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=noise.device).expand(batch, -1, -1).clone()
    correction[:, 2, 2] = torch.sign(torch.linalg.det(u @ vh))
    return target.unsqueeze(0) @ (u @ correction @ vh)


def signed_local_volume(x):
    """SO(3)-invariant, reflection-odd local C-alpha feature [B, L, 1]."""
    length = x.shape[1]
    ids = torch.arange(length, device=x.device)
    left = x[:, (ids - 1).clamp(min=0)] - x
    right = x[:, (ids + 1).clamp(max=length - 1)] - x
    far = x[:, (ids + 2).clamp(max=length - 1)] - x
    numerator = (torch.cross(left, right, dim=-1) * far).sum(-1, keepdim=True)
    denominator = (left.norm(dim=-1, keepdim=True) *
                   right.norm(dim=-1, keepdim=True) *
                   far.norm(dim=-1, keepdim=True)).clamp_min(1e-4)
    return numerator / denominator


class EGNNLayer(nn.Module):
    def __init__(self, hidden, use_signed=False):
        super().__init__()
        self.use_signed = use_signed
        self.edge = nn.Sequential(nn.Linear(2 * hidden + 2, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(2 * hidden + int(use_signed), hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden))

    def forward(self, h, x, edge_mask, separation):
        b, length, _ = x.shape
        delta = x[:, :, None, :] - x[:, None, :, :]
        distance2 = delta.square().sum(-1, keepdim=True)
        hi = h[:, :, None, :].expand(-1, -1, length, -1)
        hj = h[:, None, :, :].expand(-1, length, -1, -1)
        m = self.edge(torch.cat([hi, hj, distance2, separation], dim=-1))
        m = m * edge_mask
        node_features = [h, m.sum(2) / (length - 1)]
        if self.use_signed:
            node_features.append(signed_local_volume(x))
        h = h + self.node(torch.cat(node_features, dim=-1))
        return h, delta, m


class FlowEGNN(nn.Module):
    def __init__(self, length, hidden=64, layers=3, use_signed=False):
        super().__init__()
        self.aa = nn.Embedding(len(ALPHABET), hidden)
        self.position = nn.Embedding(length, hidden)
        self.time = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.layers = nn.ModuleList([EGNNLayer(hidden, use_signed) for _ in range(layers)])
        self.vector = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        # Cross products are SO(3) equivariant but change parity under reflection.
        # They let the flow distinguish a protein from its mirror image.
        self.chiral = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 3))
        nn.init.zeros_(self.vector[-1].weight)
        nn.init.zeros_(self.vector[-1].bias)
        nn.init.zeros_(self.chiral[-1].weight)
        nn.init.zeros_(self.chiral[-1].bias)
        ids = torch.arange(length)
        self.register_buffer("positions", ids)
        self.register_buffer("separation", (ids[:, None] - ids[None, :]).abs().float()[None, :, :, None] / length)
        self.register_buffer("edge_mask", (ids[:, None] != ids[None, :]).float()[None, :, :, None])

    def forward(self, x, t, sequence_ids):
        batch, length, _ = x.shape
        t = t.reshape(batch, 1)
        time_features = torch.cat([t, torch.sin(2 * math.pi * t), torch.cos(2 * math.pi * t)], -1)
        h = self.aa(sequence_ids)[None] + self.position(self.positions)[None] + self.time(time_features)[:, None]
        h = h.expand(batch, -1, -1)
        for layer in self.layers:
            h, delta, m = layer(h, x, self.edge_mask, self.separation.expand(batch, -1, -1, -1))
        weights = self.vector(m) * self.edge_mask
        velocity = (delta * weights).sum(2) / (length - 1)
        coefficients = self.chiral(h)
        for channel, offset in enumerate((1, 2, 4)):
            previous = self.positions.sub(offset).clamp(min=0)
            following = self.positions.add(offset).clamp(max=length - 1)
            left = x[:, previous] - x
            right = x[:, following] - x
            velocity = velocity + coefficients[:, :, channel, None] * torch.cross(left, right, dim=-1)
        return velocity - velocity.mean(1, keepdim=True)


def kabsch_rmsd(pred, truth):
    covariance = pred.T @ truth
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=pred.device)
    correction[-1, -1] = torch.sign(torch.linalg.det(u @ vh))
    aligned = pred @ (u @ correction @ vh)
    return float(torch.sqrt((aligned - truth).square().sum(-1).mean()))


@torch.no_grad()
def generate(model, ids, length, ode_steps, device, seed):
    torch.manual_seed(seed)
    x = torch.randn(1, length, 3, device=device)
    x = x - x.mean(1, keepdim=True)
    model.eval()
    dt = 1 / ode_steps
    for step in range(ode_steps):
        t = torch.full((1,), step * dt, device=device)
        v1 = model(x, t, ids)
        v2 = model(x + dt * v1, t + dt, ids)
        x = x + 0.5 * dt * (v1 + v2)
    return x[0]


def write_ca_pdb(path, sequence, coords):
    from Bio.SeqUtils import seq3
    with open(path, "w") as handle:
        for i, (aa, point) in enumerate(zip(sequence, coords), 1):
            x, y, z = point.tolist()
            handle.write(f"ATOM   {i:4d}  CA  {seq3(aa).upper():3s} A{i:4d}    {x:8.3f}{y:8.3f}{z:8.3f}{1.0:6.2f}{0.0:6.2f}           C\n")
        handle.write("END\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cif", type=Path, required=True)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--out", type=Path, default=Path("run_1crn"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ode-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--coupling", choices=("independent", "aligned"), default="aligned")
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--signed-input", action="store_true")
    parser.add_argument("--chirality-loss", type=float, default=0.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sequence, target_cpu, scale = load_protein(args.cif, args.chain)
    target = target_cpu.to(device)
    ids = torch.tensor([ALPHABET.index(a) for a in sequence], device=device)
    model = FlowEGNN(len(sequence), use_signed=args.signed_input).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    previous_steps = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint["sequence"] != sequence:
            raise ValueError("Checkpoint sequence does not match input structure")
        if checkpoint.get("coupling", "independent") != args.coupling:
            raise ValueError("Checkpoint coupling does not match --coupling")
        if checkpoint.get("signed_input", False) != args.signed_input:
            raise ValueError("Checkpoint signed input does not match --signed-input")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
            for group in optimizer.param_groups:
                group["lr"] = args.lr
        previous_steps = checkpoint.get("steps", 0)
    history = []
    for step in range(previous_steps + 1, previous_steps + args.steps + 1):
        model.train()
        x0 = torch.randn(args.batch, len(sequence), 3, device=device)
        x0 = x0 - x0.mean(1, keepdim=True)
        if args.coupling == "aligned":
            x1 = align_target_to_noise(target, x0)
        else:
            x1 = target[None] @ random_rotations(args.batch, device)
        t = torch.rand(args.batch, device=device)
        xt = (1 - t[:, None, None]) * x0 + t[:, None, None] * x1
        true_velocity = x1 - x0
        predicted = model(xt, t, ids)
        flow_loss = F.mse_loss(predicted, true_velocity)
        if args.chirality_loss:
            estimated_target = xt + (1 - t[:, None, None]) * predicted
            local_error = (signed_local_volume(estimated_target) - signed_local_volume(x1)).square()
            chiral_loss = (local_error * t[:, None, None].square()).mean()
            loss = flow_loss + args.chirality_loss * chiral_loss
        else:
            loss = flow_loss
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == previous_steps + 1 or step % 100 == 0 or step == previous_steps + args.steps:
            value = float(loss.detach())
            history.append({"step": step, "total_loss": value, "flow_mse": float(flow_loss.detach())})
            print(f"step {step:5d}  total_loss {value:.5f}  flow_mse {float(flow_loss.detach()):.5f}", flush=True)
    mirror = target * torch.tensor([-1., 1., 1.], device=device)
    evaluations = []
    generated = None
    for sample_index in range(args.eval_seeds):
        sample = generate(model, ids, len(sequence), args.ode_steps, device, args.seed + 1 + sample_index)
        if generated is None:
            generated = sample
        evaluations.append({"seed": args.seed + 1 + sample_index,
                            "target_rmsd_A": kabsch_rmsd(sample, target) * scale,
                            "mirror_rmsd_A": kabsch_rmsd(sample, mirror) * scale})
    rmsd = evaluations[0]["target_rmsd_A"]
    mirror_rmsd = evaluations[0]["mirror_rmsd_A"]
    generated_angstrom = generated.cpu() * scale
    target_angstrom = target_cpu * scale
    generated_bonds = torch.linalg.vector_norm(generated_angstrom[1:] - generated_angstrom[:-1], dim=-1)
    target_bonds = torch.linalg.vector_norm(target_angstrom[1:] - target_angstrom[:-1], dim=-1)
    metrics = {"pdb": args.cif.stem, "chain": args.chain, "sequence": sequence,
               "residues": len(sequence), "device": str(device), "steps": previous_steps + args.steps,
               "coupling": args.coupling, "evaluations": evaluations,
               "signed_input": args.signed_input, "chirality_loss_weight": args.chirality_loss,
               "median_target_rmsd_A": float(np.median([e["target_rmsd_A"] for e in evaluations])),
               "target_closer_count": sum(e["target_rmsd_A"] < e["mirror_rmsd_A"] for e in evaluations),
               "scale_angstrom": scale, "aligned_ca_rmsd_angstrom": rmsd,
               "mirror_aligned_ca_rmsd_angstrom": mirror_rmsd,
               "generated_adjacent_ca_mean_angstrom": float(generated_bonds.mean()),
               "target_adjacent_ca_mean_angstrom": float(target_bonds.mean()),
               "history": history}
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "steps": previous_steps + args.steps, "sequence": sequence, "scale": scale,
                "coupling": args.coupling, "signed_input": args.signed_input,
                "chirality_loss_weight": args.chirality_loss}, args.out / "checkpoint.pt")
    (args.out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    write_ca_pdb(args.out / "generated_ca.pdb", sequence, generated_angstrom)
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("history", "sequence")}, indent=2))


if __name__ == "__main__":
    main()
