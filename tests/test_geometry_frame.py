import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))

from geometry_frame import FrameRefinement, GeometryFrame  # noqa: E402
from geometry_metrics import fixed_correspondence_tm_score  # noqa: E402
from geometry_targets import geometry_losses, target_geometry  # noqa: E402
from geometry_train import intermediate_loss  # noqa: E402
from pair_frame import PairFrame  # noqa: E402


def proper_rotation():
    q, _ = torch.linalg.qr(torch.randn(3, 3))
    if torch.linalg.det(q) < 0:
        q[:, -1] = -q[:, -1]
    return q


class GeometryFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(19)

    def test_geometry_heads_shapes_symmetry_and_gradient(self):
        model = GeometryFrame(hidden=32, pair_dim=16, heads=4, layers=2, refine_steps=2)
        ca, rotation, distance, aux = model.forward_with_aux(torch.randn(7, 320),
                                                              torch.randint(0, 20, (7,)))
        self.assertEqual((ca.shape, rotation.shape, distance.shape),
                         ((7, 3), (7, 3, 3), (7, 7)))
        self.assertEqual(aux["pair"].shape, (7, 7, 16))
        self.assertEqual(aux["distogram_logits"].shape, (7, 7, 11))
        self.assertEqual(aux["contact_logits"].shape, (7, 7))
        self.assertEqual(aux["relative_rotation"].shape, (7, 7, 3, 3))
        self.assertEqual(aux["relative_direction"].shape, (7, 7, 3))
        self.assertEqual(len(aux["intermediate"]), 2)
        self.assertTrue(torch.allclose(aux["distogram_logits"],
                                       aux["distogram_logits"].transpose(0, 1), atol=1e-7))
        self.assertTrue(torch.allclose(aux["contact_logits"], aux["contact_logits"].T, atol=1e-7))
        self.assertTrue(torch.all(torch.linalg.det(rotation) > 0.99))
        objective = (ca - torch.randn_like(ca)).square().mean()
        objective = objective + aux["distogram_logits"].square().mean()
        objective = objective + aux["contact_logits"].square().mean()
        objective = objective + aux["relative_direction"].square().mean()
        objective.backward()
        self.assertGreater(model.distogram_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.contact_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.orientation_head[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.refinement.node[-1].weight.grad.abs().sum().item(), 0)

    def test_initial_structure_matches_pair_frame(self):
        torch.manual_seed(23)
        baseline = PairFrame(hidden=32, pair_dim=16, heads=4, layers=2)
        geometry = GeometryFrame(hidden=32, pair_dim=16, heads=4, layers=2)
        geometry.load_state_dict(baseline.state_dict(), strict=False)
        features, aa = torch.randn(8, 320), torch.randint(0, 20, (8,))
        for original, augmented in zip(baseline(features, aa), geometry(features, aa)):
            self.assertTrue(torch.allclose(original, augmented, atol=1e-6))

    def test_refinement_rigid_equivariance_and_valid_rotations(self):
        module = FrameRefinement(pair_dim=16, hidden=32, steps=4)
        with torch.no_grad():
            module.node[-1].weight.normal_(std=0.1)
            module.node[-1].bias.normal_(std=0.1)
        ca = torch.randn(6, 3)
        rotation = torch.stack([proper_rotation() for _ in range(6)])
        pair = torch.randn(6, 6, 16)
        q, shift = proper_rotation(), torch.randn(3)
        new_ca, new_rotation, _ = module(ca, rotation, pair)
        transformed_ca, transformed_rotation, _ = module(ca @ q.T + shift,
                                                           q @ rotation, pair)
        self.assertTrue(torch.allclose(transformed_ca, new_ca @ q.T + shift, atol=2e-5))
        self.assertTrue(torch.allclose(transformed_rotation, q @ new_rotation, atol=2e-5))
        self.assertTrue(torch.allclose(new_rotation.transpose(-1, -2) @ new_rotation,
                                       torch.eye(3).expand(6, 3, 3), atol=2e-5))
        self.assertTrue(torch.all(torch.linalg.det(new_rotation) > 0.999))

    def test_geometry_targets_masks_missing_and_chain_break(self):
        length = 6
        ca = torch.stack((torch.arange(length).float() * 0.38,
                          torch.zeros(length), torch.zeros(length)), dim=-1)
        atoms = ca[:, None, :].expand(length, 3, 3).clone()
        atoms[:, 0, 0] -= 0.12
        atoms[:, 2, 0] += 0.12
        protein = dict(ca=ca, rot=torch.eye(3).expand(length, 3, 3).clone(), atoms=atoms)
        valid = torch.tensor([1, 1, 0, 1, 1, 1], dtype=torch.bool)
        breaks = torch.tensor([0, 0, 1, 0, 0], dtype=torch.bool)
        target = target_geometry(protein, valid_residue=valid,
                                 chain_break_after=breaks, contact_min_separation=2)
        self.assertFalse(target["distance_mask"][0, 3])
        self.assertFalse(target["distance_mask"][0, 2])
        self.assertFalse(target["distance_mask"][1, 0])
        self.assertTrue(target["distance_mask"][3, 5])
        self.assertTrue(target["contact_mask"][3, 5])
        self.assertTrue(torch.equal(target["distogram_bins"], target["distogram_bins"].T))
        self.assertTrue(torch.allclose(target["relative_direction"][3, 5],
                                       -target["relative_direction"][5, 3]))
        broken_atoms = atoms.clone()
        broken_atoms[2, 2, 0] += 1.0
        inferred = target_geometry(dict(ca=ca, rot=protein["rot"], atoms=broken_atoms))
        self.assertTrue(bool(inferred["chain_break_after"][2]))
        self.assertFalse(bool(inferred["distance_mask"][0, 3]))
        aux = dict(distogram_logits=torch.randn(length, length, 11, requires_grad=True),
                   contact_logits=torch.randn(length, length, requires_grad=True),
                   relative_rotation=torch.eye(3).expand(length, length, 3, 3).clone().requires_grad_(),
                   relative_direction=torch.randn(length, length, 3, requires_grad=True))
        losses = geometry_losses(aux, target)
        sum(losses.values()).backward()
        self.assertTrue(all(torch.isfinite(loss) for loss in losses.values()))
        self.assertEqual(float(aux["distogram_logits"].grad[0, 3].abs().sum()), 0)

    def test_fixed_correspondence_tm_score_rigid_invariance(self):
        ca = torch.randn(25, 3)
        q, shift = proper_rotation(), torch.randn(3)
        self.assertAlmostEqual(fixed_correspondence_tm_score(ca.numpy(), ca.numpy()), 1.0, places=6)
        transformed = ca @ q.T + shift
        self.assertAlmostEqual(fixed_correspondence_tm_score(transformed.numpy(), ca.numpy()),
                               1.0, places=5)

    def test_optional_intermediate_structure_loss_has_gradient(self):
        target_ca = torch.randn(5, 3)
        rotation = torch.eye(3).expand(5, 3, 3).clone()
        aa = torch.randint(0, 20, (5,))
        local = torch.randn(20, 3, 3) * 0.01
        atoms = target_ca[:, None] + torch.einsum("lai,lji->laj", local[aa], rotation)
        protein = dict(ca=target_ca, rot=rotation, atoms=atoms, aa=aa)
        predicted = torch.randn(5, 3, requires_grad=True)
        loss = intermediate_loss([(predicted, rotation)], protein, local)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(predicted.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
