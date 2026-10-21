"""Freeze ESM-2 embeddings and pack backbone frames into resumable shards."""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from Bio.PDB import MMCIFParser, PDBParser
from Bio.SeqUtils import seq1
from transformers import AutoTokenizer, EsmModel


ALPHABET = "ACDEFGHIKLMNPQRSTVWY"


def fingerprint(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_structure(row):
    path = Path(row["path"])
    if not path.is_absolute():
        path = (Path(__file__).resolve().parent.parent / "outputs" / path).resolve()
    parser = PDBParser(QUIET=True) if path.suffix.lower() == ".pdb" else MMCIFParser(QUIET=True)
    chain = parser.get_structure("protein", str(path))[0][row["chain"]]
    residues = [residue for residue in chain if residue.id[0] == " " and
                all(atom in residue for atom in ("N", "CA", "C"))]
    sequence = "".join(seq1(residue.resname, custom_map={"UNK": "X"}) for residue in residues)
    if sequence != row["sequence"]:
        raise ValueError(f"structure/manifest sequence mismatch: {row['id']}:{row['chain']}")
    atoms = torch.tensor([[[float(x) for x in residue[name].coord]
                           for name in ("N", "CA", "C")] for residue in residues])
    ca = atoms[:, 1]
    e1 = torch.nn.functional.normalize(atoms[:, 2] - ca, dim=-1)
    v2 = atoms[:, 0] - ca
    e2 = torch.nn.functional.normalize(v2 - (v2 * e1).sum(-1, keepdim=True) * e1, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    rotation = torch.stack((e1, e2, e3), dim=-1)
    center = ca.mean(0)
    atoms = (atoms - center) / 10
    ca = (ca - center) / 10
    breaks = [False]
    for left, right in zip(residues, residues[1:]):
        cn = float(torch.tensor(left["C"].coord - right["N"].coord).norm())
        breaks.append(right.id[1] != left.id[1] + 1 or cn > 2.2)
    segments = torch.tensor(breaks, dtype=torch.long).cumsum(0)
    return dict(aa=torch.tensor([ALPHABET.index(letter) for letter in sequence]),
                ca=ca, rot=rotation, atoms=atoms, segments=segments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="facebook/esm2_t6_8M_UR50D")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1 or args.shard_size < 1:
        parser.error("batch-size and shard-size must be positive")
    rows = json.loads(args.manifest.read_text())["proteins"]
    if any(row["split"] not in ("train", "val", "test") for row in rows):
        parser.error("manifest must be finalized; candidate rows are not trainable")
    manifest_sha = fingerprint(args.manifest)
    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.json"
    index = json.loads(index_path.read_text()) if index_path.exists() else {
        "manifest_sha256": manifest_sha, "model": args.model,
        "shard_size": args.shard_size, "count": 0, "shards": []}
    if (index["manifest_sha256"] != manifest_sha or index["model"] != args.model or
            index["shard_size"] != args.shard_size):
        raise ValueError("Output directory contains a different manifest/model/shard size")
    if any(not (args.out / shard["file"]).is_file() for shard in index["shards"]):
        raise ValueError("index references a missing shard")
    if any(fingerprint(args.out / shard["file"]) != shard["sha256"]
           for shard in index["shards"]):
        raise ValueError("cached shard checksum mismatch")
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = EsmModel.from_pretrained(args.model, add_pooling_layer=False).to(device).eval()
    for start in range(index["count"], len(rows), args.shard_size):
        chunk = rows[start:start + args.shard_size]
        records = []
        for offset in range(0, len(chunk), args.batch_size):
            batch = chunk[offset:offset + args.batch_size]
            tokenized = tokenizer([row["sequence"] for row in batch], padding=True,
                                  return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**tokenized).last_hidden_state
            for row, encoded in zip(batch, hidden):
                length = len(row["sequence"])
                record = dict(id=row["id"], chain=row["chain"], split=row["split"],
                              sequence=row["sequence"], source=row.get("source", "PDB"),
                              embed=encoded[1:length + 1].cpu().half(),
                              **parse_structure(row))
                records.append(record)
        name = f"shard_{start // args.shard_size:05d}.pt"
        temporary = args.out / (name + ".tmp")
        torch.save(records, temporary)
        temporary.replace(args.out / name)
        index["shards"].append(dict(file=name, count=len(records),
                                    splits={split: sum(r["split"] == split for r in records)
                                            for split in ("train", "val", "test")},
                                    sha256=fingerprint(args.out / name)))
        index["count"] += len(records)
        index_tmp = index_path.with_suffix(".json.tmp")
        index_tmp.write_text(json.dumps(index, indent=2))
        index_tmp.replace(index_path)
        print(json.dumps({"packed": index["count"], "total": len(rows), "shard": name}), flush=True)
    if index["count"] != len(rows):
        raise ValueError("incomplete cache")


if __name__ == "__main__":
    main()
