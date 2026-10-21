import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "outputs"))

from direct_frame import DirectFrame, losses  # noqa: E402
from pair_frame import PairFrame  # noqa: E402
from pair_train import consistency_loss, seed_everything, update_best, validation_score  # noqa: E402


class PairFrameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def model_and_input(self, length=8):
        model = PairFrame(hidden=32, pair_dim=16, heads=4, layers=2)
        features = torch.randn(length, 320)
        aa = torch.randint(0, 20, (length,))
        return model, features, aa

    def test_forward_shapes_rotation_and_distance_symmetry(self):
        model, features, aa = self.model_and_input()
        ca, rotation, distance = model(features, aa)
        self.assertEqual(ca.shape, (8, 3))
        self.assertEqual(rotation.shape, (8, 3, 3))
        self.assertEqual(distance.shape, (8, 8))
        self.assertTrue(torch.allclose(distance, distance.T, atol=1e-7))
        self.assertTrue(torch.equal(torch.diag(distance), torch.zeros(8)))
        self.assertTrue(torch.allclose(rotation.transpose(-1, -2) @ rotation,
                                       torch.eye(3).expand(8, 3, 3), atol=1e-5))
        self.assertTrue(torch.all(torch.linalg.det(rotation) > 0.99))

    def test_default_pair_representation_shape_and_model_size(self):
        model = PairFrame()
        captured = []
        handle = model.pair_init.register_forward_hook(lambda _module, _inputs, output: captured.append(output.shape))
        model(torch.randn(6, 320), torch.randint(0, 20, (6,)))
        handle.remove()
        self.assertEqual(captured, [torch.Size((6, 6, 64))])
        self.assertLess(sum(parameter.numel() for parameter in model.parameters()), 1_200_000)

    def test_backward_reaches_pair_initialization_and_updates(self):
        model, features, aa = self.model_and_input()
        ca, rotation, distance = model(features, aa)
        objective = ca.square().mean() + rotation.square().mean() + distance.square().mean()
        objective.backward()
        self.assertIsNotNone(model.pair_init[0].weight.grad)
        self.assertGreater(model.pair_init[0].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].pair_update[-1].weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.blocks[0].bias.weight.grad.abs().sum().item(), 0)
        self.assertGreater(model.dist[-1].weight.grad.abs().sum().item(), 0)

    def test_original_loss_and_optional_consistency(self):
        model, features, aa = self.model_and_input()
        with torch.no_grad():
            target_ca, target_rot, _ = model(features, aa)
        local = torch.randn(20, 3, 3) * 0.01
        atoms = target_ca[:, None, :] + torch.einsum("lai,lji->laj", local[aa], target_rot)
        protein = dict(embed=features, aa=aa, ca=target_ca, rot=target_rot, atoms=atoms)
        ca, rot, distance = model(features, aa)
        base, parts = losses(lambda _features, _aa: (ca, rot, distance), protein, local)
        consistency = consistency_loss(ca, distance)
        (base + 0.5 * consistency).backward()
        self.assertTrue(torch.isfinite(base))
        self.assertTrue(torch.isfinite(consistency))
        self.assertIn("pair", parts)
        self.assertGreater(model.dist[-1].weight.grad.abs().sum().item(), 0)

    def test_baseline_interface_and_selection_score(self):
        model = DirectFrame()
        features = torch.randn(5, 320)
        aa = torch.randint(0, 20, (5,))
        ca, rotation, distance = model(features, aa)
        self.assertEqual((ca.shape, rotation.shape, distance.shape),
                         ((5, 3), (5, 3, 3), (5, 5)))
        self.assertLess(validation_score(dict(ca_rmsd_A=8, lddt_ca=0.4)),
                        validation_score(dict(ca_rmsd_A=9, lddt_ca=0.3)))

    def test_fixed_seed_repeats_initial_predictions(self):
        seed_everything(11)
        first = PairFrame(hidden=32, pair_dim=16, heads=4, layers=1)
        features = torch.randn(5, 320)
        aa = torch.randint(0, 20, (5,))
        prediction = first(features, aa)
        seed_everything(11)
        second = PairFrame(hidden=32, pair_dim=16, heads=4, layers=1)
        repeated_features = torch.randn(5, 320)
        repeated_aa = torch.randint(0, 20, (5,))
        repeated = second(repeated_features, repeated_aa)
        self.assertTrue(torch.equal(features, repeated_features))
        self.assertTrue(torch.equal(aa, repeated_aa))
        for left, right in zip(prediction, repeated):
            self.assertTrue(torch.equal(left, right))

    def test_best_selection_and_early_stopping_count(self):
        best, stale = float("inf"), 0
        improved_states = []
        for score in (1.0, 1.0005, 1.1, 0.998, 0.999):
            best, stale, improved = update_best(score, best, stale, min_delta=0.001)
            improved_states.append((improved, stale))
        self.assertEqual(improved_states, [(True, 0), (False, 1), (False, 2),
                                           (True, 0), (False, 1)])
        self.assertEqual(best, 0.998)


if __name__ == "__main__":
    unittest.main()
