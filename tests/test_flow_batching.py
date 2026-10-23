import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))

from backbone_atoms import reconstruct_backbone  # noqa: E402
from flow_batching import collate, train_batches  # noqa: E402
from flow_model_large import SequenceFrameFlow  # noqa: E402
from train_flow_large import batch_flow_loss  # noqa: E402


def record(length, seed):
    generator = torch.Generator().manual_seed(seed)
    ca = torch.randn(length, 3, generator=generator)
    rot = torch.eye(3).expand(length, 3, 3).clone()
    return dict(embed=torch.randn(length, 16, generator=generator).half(),
                aa=torch.randint(0, 20, (length,), generator=generator),
                ca=ca, rot=rot,
                atoms=reconstruct_backbone(ca, rot),
                atom_mask=torch.ones(length, 4, dtype=torch.bool),
                segments=torch.zeros(length, dtype=torch.long))


class FlowBatchingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(73)

    def test_masked_batch_matches_unpadded_forward(self):
        model = SequenceFrameFlow(esm_dim=16, hidden=32, pair_dim=16,
                                  heads=4, pair_layers=2, geometric_layers=1).eval()
        first, second = record(5, 1), record(8, 2)
        batch = collate([first, second])
        time = torch.tensor([0.3, 0.7])
        with torch.no_grad():
            batched = model(batch["embed"].float(), batch["aa"], batch["ca"],
                            batch["rot"], time, batch["mask"])
            alone = model(first["embed"].float(), first["aa"], first["ca"],
                          first["rot"], time[0])
        for batched_output, alone_output in zip(batched, alone):
            relevant = batched_output[0, :5, :5] if batched_output.ndim == 4 else batched_output[0, :5]
            self.assertTrue(torch.allclose(relevant, alone_output, atol=2e-5, rtol=2e-5))
            self.assertTrue(torch.isfinite(batched_output).all())
        self.assertTrue(torch.equal(batched[0][0, 5:], torch.zeros_like(batched[0][0, 5:])))
        self.assertTrue(torch.equal(batched[2][0, 5:], torch.zeros_like(batched[2][0, 5:])))
        self.assertTrue(torch.allclose(batched[2], batched[2].transpose(1, 2)))

    def test_batched_loss_backward(self):
        model = SequenceFrameFlow(esm_dim=16, hidden=32, pair_dim=16,
                                  heads=4, pair_layers=2, geometric_layers=1)
        batch = collate([record(5, 3), record(7, 4)])
        batch["embed"] = batch["embed"].float()
        loss, parts = batch_flow_loss(model, batch)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(value >= 0 for value in parts.values()))
        loss.backward()
        self.assertGreater(model.velocity[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.distogram[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.pair_init[0].weight.grad.abs().sum().item(), 0)

    def test_shard_iterator_keeps_train_only_and_all_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for index, length in enumerate((5, 8, 6, 9, 7)):
                row = record(length, index)
                row["split"] = "val" if index == 3 else "train"
                rows.append(row)
            torch.save(rows, root / "part.pt")
            index = {"shards": [{"file": "part.pt", "splits": {"train": 4, "val": 1}}]}
            batches = list(train_batches(root, index, 2, seed=11, epoch=0))
        self.assertEqual(sum(batch["mask"].sum().item() for batch in batches),
                         sum((5, 8, 6, 7)))
        self.assertEqual(sum(len(batch["aa"]) for batch in batches), 4)
        self.assertTrue(all(batch["mask"].sum(1).min().item() >= 5 for batch in batches))


if __name__ == "__main__":
    unittest.main()
