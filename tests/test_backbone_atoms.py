import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))

from backbone_atoms import peptide_geometry_loss, reconstruct_backbone  # noqa: E402
from train_flow_large import score_backbone  # noqa: E402


class BackboneAtomsTests(unittest.TestCase):
    def test_shape_gradients_and_se3_equivariance(self):
        torch.manual_seed(4)
        ca = torch.randn(2, 5, 3, dtype=torch.float64, requires_grad=True)
        rotation = torch.eye(3, dtype=torch.float64).expand(2, 5, 3, 3).clone()
        atoms = reconstruct_backbone(ca, rotation)
        self.assertEqual(atoms.shape, (2, 5, 4, 3))
        self.assertTrue(torch.equal(atoms[:, :, 1], ca))
        q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
        if torch.linalg.det(q) < 0:
            q[:, -1] *= -1
        shift = torch.randn(3, dtype=torch.float64)
        moved = reconstruct_backbone(ca @ q + shift, q.T @ rotation)
        self.assertTrue(torch.allclose(moved, atoms @ q + shift, atol=1e-10))
        atoms.square().sum().backward()
        self.assertTrue(torch.isfinite(ca.grad).all())
        self.assertGreater(ca.grad.abs().sum().item(), 0)

    def test_segment_mask_and_scoring(self):
        ca = torch.tensor([[0., 0., 0.], [.4, 0., 0.], [.8, 0., 0.],
                           [1.2, 0., 0.], [1.6, 0., 0.]])
        rotation = torch.eye(3).expand(5, 3, 3).clone()
        atoms = reconstruct_backbone(ca, rotation)
        segments = torch.tensor([0, 0, 1, 1, 1])
        mask = torch.tensor([True, True, True, True, False])
        modified = atoms.clone()
        modified[2, 0] += 10
        modified[4, 0] += 10
        bond, angle = peptide_geometry_loss(atoms[None], modified[None],
                                            segments[None], mask[None])
        self.assertEqual(float(bond), 0)
        self.assertEqual(float(angle), 0)
        record = dict(ca=ca, atoms=atoms, atom_mask=torch.ones(5, 4, dtype=torch.bool),
                      segments=segments)
        scores = score_backbone(ca, rotation, record)
        self.assertLess(scores["backbone_atom_mae_A"], 1e-5)
        self.assertLess(scores["peptide_bond_mae_A"], 1e-5)


if __name__ == "__main__":
    unittest.main()
