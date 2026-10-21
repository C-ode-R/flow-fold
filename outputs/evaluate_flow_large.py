"""Evaluate one validation-selected Flow checkpoint on the held-out test split."""
import argparse
import json
from pathlib import Path

import torch

from flow_model_large import SequenceFrameFlow
from train_flow_large import evaluate, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    index_path = args.cache / "index.json"
    index = json.loads(index_path.read_text())
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                          "cpu" if args.device == "auto" else args.device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if checkpoint["index_sha256"] != sha256(index_path):
        raise ValueError("Checkpoint and feature cache differ")
    model = SequenceFrameFlow(**checkpoint["config"]).to(device).eval()
    model.load_state_dict(checkpoint["model"])
    summary, rows = evaluate(model, args.cache, index, device,
                             checkpoint["ode_steps"], checkpoint["seed"], split="test")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(dict(checkpoint=str(args.checkpoint),
                                        selected_epoch=checkpoint["epoch"],
                                        summary=summary, rows=rows), indent=2))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
