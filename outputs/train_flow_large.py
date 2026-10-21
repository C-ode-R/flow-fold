"""Train sequence-conditioned SE(3) flow from frozen ESM/frame shards.

Run from outputs/. Train and validation only are loaded. Test is never read.
Checkpointing occurs at epoch boundaries; resume restarts the next epoch.
"""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from flow_fold import kabsch_rmsd, random_rotations
from flow_model_large import SequenceFrameFlow
from geometry_frame import DISTANCE_EDGES_A
from geometry_metrics import fixed_correspondence_tm_score
from se3_flow import so3_exp, so3_log


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def on_device(record, device):
    return {key: (value.to(device=device, dtype=torch.float32) if key in
                  ("embed", "ca", "rot", "atoms") else value.to(device) if
                  isinstance(value, torch.Tensor) else value)
            for key, value in record.items()}


def aligned_target(ca, rotation, source):
    # Row-vector Kabsch, constrained to det +1. Never reflect the target.
    covariance = ca.T @ source
    u, _, vh = torch.linalg.svd(covariance)
    correction = torch.eye(3, device=ca.device, dtype=ca.dtype)
    correction[-1, -1] = torch.sign(torch.linalg.det(u @ vh))
    row_rotation = u @ correction @ vh
    return ca @ row_rotation, row_rotation.T @ rotation


def flow_loss(model, record, distance_weight=0.05, fape_weight=0.2):
    ca1, rotation1 = record["ca"], record["rot"]
    length = len(ca1)
    ca0 = torch.randn_like(ca1)
    ca0 = ca0 - ca0.mean(0, keepdim=True)
    rotation0 = random_rotations(length, ca1.device)
    ca1, rotation1 = aligned_target(ca1, rotation1, ca0)
    omega = so3_log(rotation0.transpose(-1, -2) @ rotation1)
    time = 0.02 + 0.96 * torch.rand((), device=ca1.device)
    ca_t = (1 - time) * ca0 + time * ca1
    rotation_t = rotation0 @ so3_exp(time * omega)
    target_global_v = ca1 - ca0
    target_local_v = torch.einsum("lj,lji->li", target_global_v, rotation_t)
    with torch.autocast(device_type=ca1.device.type, dtype=torch.bfloat16,
                        enabled=ca1.device.type == "cuda"):
        predicted_v, predicted_w, dist_logits = model(record["embed"], record["aa"],
                                                      ca_t, rotation_t, time)
    predicted_v, predicted_w = predicted_v.float(), predicted_w.float()
    position = F.mse_loss(predicted_v, target_local_v)
    angular = F.mse_loss(predicted_w, omega)
    distance_a = torch.cdist(ca1, ca1) * 10
    bins = torch.bucketize(distance_a, distance_a.new_tensor(DISTANCE_EDGES_A))
    upper = torch.triu(torch.ones(length, length, dtype=torch.bool, device=ca1.device), diagonal=1)
    same_segment = record["segments"][:, None] == record["segments"][None]
    pair_mask = upper & same_segment
    distogram = F.cross_entropy(dist_logits[pair_mask], bins[pair_mask]) if pair_mask.any() else position * 0
    estimated_ca = ca_t + (1 - time) * torch.einsum("li,lji->lj", predicted_v, rotation_t)
    estimated_rotation = rotation_t @ so3_exp((1 - time) * predicted_w)
    predicted_relative = torch.einsum("ijd,idk->ijk", estimated_ca[None] - estimated_ca[:, None],
                                      estimated_rotation)
    target_relative = torch.einsum("ijd,idk->ijk", ca1[None] - ca1[:, None], rotation1)
    fape = ((predicted_relative - target_relative).norm(dim=-1)[pair_mask]).clamp(max=1).mean() if pair_mask.any() else position * 0
    total = position + angular + distance_weight * distogram + fape_weight * fape
    return total, dict(velocity=float(position.detach()), angular=float(angular.detach()),
                       distogram=float(distogram.detach()), fape=float(fape.detach()))


@torch.no_grad()
def generate(model, record, steps, seed):
    device = record["ca"].device
    with torch.random.fork_rng(devices=[device.index if device.index is not None else torch.cuda.current_device()]
                               if device.type == "cuda" else []):
        torch.manual_seed(seed)
        length = len(record["aa"])
        ca = torch.randn(length, 3, device=device)
        ca -= ca.mean(0, keepdim=True)
        rotation = random_rotations(length, device)
        for step in range(steps):
            time = torch.tensor(step / steps, device=device)
            velocity, angular, _ = model(record["embed"], record["aa"], ca, rotation, time)
            ca = ca + torch.einsum("li,lji->lj", velocity, rotation) / steps
            rotation = rotation @ so3_exp(angular / steps)
    return ca, rotation


def score_one(predicted, target):
    x, y = predicted * 10, target * 10
    length = len(x)
    dx, dy = torch.cdist(x, x), torch.cdist(y, y)
    index = torch.arange(length, device=x.device)
    separation = (index[:, None] - index[None]).abs()
    local = torch.triu((dy < 15) & (separation > 1), diagonal=1)
    lddt = sum(float(((dx - dy).abs()[local] < threshold).float().mean())
                for threshold in (0.5, 1.0, 2.0, 4.0)) / 4 if local.any() else 0
    long_mask = torch.triu(separation >= 12, diagonal=1)
    top = min(length, int(long_mask.sum()))
    contact = float((dy[long_mask][torch.topk(-dx[long_mask], top).indices] < 8)
                    .float().mean()) if top else 0
    return dict(ca_rmsd_A=float(kabsch_rmsd(x - x.mean(0), y - y.mean(0))),
                lddt_ca=lddt,
                tm_score_fixed=fixed_correspondence_tm_score(x.cpu().numpy(), y.cpu().numpy()),
                contact_precision_L=contact)


def evaluate(model, cache_dir, index, device, ode_steps, seed, limit=None, split="val"):
    model.eval()
    rows = []
    for shard in index["shards"]:
        if not shard["splits"].get(split):
            continue
        for record in torch.load(cache_dir / shard["file"], map_location="cpu", weights_only=True):
            if record["split"] != split:
                continue
            record = on_device(record, device)
            generated, _ = generate(model, record, ode_steps, seed + len(rows))
            rows.append(dict(id=record["id"], **score_one(generated, record["ca"])))
            if limit and len(rows) >= limit:
                break
        if limit and len(rows) >= limit:
            break
    if not rows:
        raise ValueError(f"No {split} records in cache")
    summary = {key: float(np.median([row[key] for row in rows]))
               for key in ("ca_rmsd_A", "lddt_ca", "tm_score_fixed", "contact_precision_L")}
    summary["n"] = len(rows)
    return summary, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ode-steps", type=int, default=25)
    parser.add_argument("--val-limit", type=int, help="Development only; omit for final selection")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if min(args.epochs, args.accumulate, args.ode_steps) < 1:
        parser.error("epochs, accumulate, and ode-steps must be positive")
    index_path = args.cache / "index.json"
    index = json.loads(index_path.read_text())
    if index["count"] != sum(shard["count"] for shard in index["shards"]):
        raise ValueError("incomplete shard index")
    if not sum(shard["splits"].get("train", 0) for shard in index["shards"]):
        raise ValueError("cache has no training records")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    first = torch.load(args.cache / index["shards"][0]["file"], map_location="cpu", weights_only=True)[0]
    model = SequenceFrameFlow(esm_dim=first["embed"].shape[-1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    start_epoch, best_score = 0, float("inf")
    if args.resume:
        saved = torch.load(args.resume, map_location=device, weights_only=False)
        if saved["index_sha256"] != sha256(index_path):
            raise ValueError("resume cache index changed")
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"].cpu())
        if device.type == "cuda" and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        start_epoch, best_score = saved["epoch"], saved["best_score"]
    for epoch in range(start_epoch, args.epochs):
        model.train()
        shard_order = [shard for shard in index["shards"] if shard["splits"].get("train")]
        random.shuffle(shard_order)
        optimizer.zero_grad(set_to_none=True)
        pending, seen, sum_loss = 0, 0, 0.0
        for shard in shard_order:
            records = torch.load(args.cache / shard["file"], map_location="cpu", weights_only=True)
            train_records = [record for record in records if record["split"] == "train"]
            random.shuffle(train_records)
            for record in train_records:
                record = on_device(record, device)
                loss, parts = flow_loss(model, record)
                (loss / args.accumulate).backward()
                pending += 1
                seen += 1
                sum_loss += float(loss.detach())
                if pending == args.accumulate:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
            del records, train_records
        if pending:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        summary, rows = evaluate(model, args.cache, index, device, args.ode_steps,
                                 args.seed, args.val_limit)
        score = summary["ca_rmsd_A"] / 10 + 1 - summary["lddt_ca"]
        report = dict(epoch=epoch + 1, train_samples=seen, train_loss=sum_loss / seen,
                      val_score=score, val_summary=summary, last_train_parts=parts)
        with (args.out / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        improved = score < best_score
        best_score = min(best_score, score)
        checkpoint = dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                          config=model.config, epoch=epoch + 1, best_score=best_score,
                          seed=args.seed, ode_steps=args.ode_steps,
                          index_sha256=sha256(index_path), manifest_sha256=index["manifest_sha256"],
                          python_rng=random.getstate(), torch_rng=torch.get_rng_state(),
                          cuda_rng=torch.cuda.get_rng_state_all() if device.type == "cuda" else None)
        temporary = args.out / "last.pt.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(args.out / "last.pt")
        if improved:
            temporary = args.out / "best.pt.tmp"
            torch.save(checkpoint, temporary)
            temporary.replace(args.out / "best.pt")
            (args.out / "best_validation.json").write_text(json.dumps(
                dict(summary=summary, rows=rows, epoch=epoch + 1), indent=2))


if __name__ == "__main__":
    main()
