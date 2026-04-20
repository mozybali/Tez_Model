from __future__ import annotations

import unittest

import torch

from Model.pointnet import build_pointnet_classifier, volume_to_pointcloud
from Model.resnet3d import build_resnet3d
from Model.unet3d import build_unet3d_classifier
from Model.factory import build_model
from Utils.config import ModelConfig


class ResNet3DTests(unittest.TestCase):
    def test_resnet18_output_shape(self) -> None:
        model = build_resnet3d(depth=18, in_channels=1, base_channels=16, dropout=0.1, num_classes=1)
        inputs = torch.randn(2, 1, 64, 64, 64)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_resnet18_with_tabular_features(self) -> None:
        model = build_resnet3d(
            depth=18,
            in_channels=1,
            base_channels=16,
            dropout=0.1,
            num_classes=1,
            num_tabular_features=2,
            tabular_hidden_dim=8,
        )
        inputs = torch.randn(2, 1, 64, 64, 64)
        tabular = torch.randn(2, 2)
        logits = model(inputs, tabular_features=tabular)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_resnet34_output_shape(self) -> None:
        model = build_resnet3d(depth=34, in_channels=1, base_channels=16, dropout=0.1, num_classes=1)
        inputs = torch.randn(2, 1, 48, 48, 48)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_single_sample_no_crash(self) -> None:
        model = build_resnet3d(depth=18, in_channels=1, base_channels=16, dropout=0.0, num_classes=1)
        model.eval()
        inputs = torch.randn(1, 1, 32, 32, 32)
        with torch.no_grad():
            logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_invalid_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_resnet3d(depth=50)


class UNet3DClassifierTests(unittest.TestCase):
    def test_output_shape_without_tabular(self) -> None:
        model = build_unet3d_classifier(
            in_channels=1, base_channels=8, depth=3, channel_multiplier=2,
            dropout=0.1, num_classes=1,
        )
        inputs = torch.randn(2, 1, 32, 32, 32)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_output_shape_with_tabular(self) -> None:
        model = build_unet3d_classifier(
            in_channels=1, base_channels=8, depth=3, channel_multiplier=2,
            dropout=0.1, num_classes=1, num_tabular_features=2, tabular_hidden_dim=8,
        )
        inputs = torch.randn(2, 1, 32, 32, 32)
        tabular = torch.randn(2, 2)
        logits = model(inputs, tabular_features=tabular)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_small_input_smoke(self) -> None:
        model = build_unet3d_classifier(
            in_channels=1, base_channels=4, depth=2, channel_multiplier=2, dropout=0.0,
        )
        model.eval()
        inputs = torch.randn(1, 1, 24, 24, 24)
        with torch.no_grad():
            logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_invalid_depth_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_unet3d_classifier(depth=1)

    def test_invalid_base_channels_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_unet3d_classifier(base_channels=0)

    def test_invalid_channel_multiplier_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_unet3d_classifier(channel_multiplier=0)

    def test_invalid_dropout_raises(self) -> None:
        with self.assertRaises(ValueError):
            build_unet3d_classifier(dropout=1.5)


class PointNetClassifierTests(unittest.TestCase):
    def _make_binary_volume(self, batch: int, edge: int, density: float = 0.3) -> torch.Tensor:
        torch.manual_seed(0)
        return (torch.rand(batch, 1, edge, edge, edge) < density).float()

    def test_output_shape_without_tabular(self) -> None:
        model = build_pointnet_classifier(
            num_points=256, mlp_channels=(32, 64), global_dim=64, head_hidden_dim=32,
            dropout=0.0,
        )
        inputs = self._make_binary_volume(2, 16)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_output_shape_with_tabular(self) -> None:
        model = build_pointnet_classifier(
            num_points=256, mlp_channels=(32, 64), global_dim=64, head_hidden_dim=32,
            dropout=0.0, num_tabular_features=2, tabular_hidden_dim=8,
        )
        inputs = self._make_binary_volume(2, 16)
        tabular = torch.randn(2, 2)
        logits = model(inputs, tabular_features=tabular)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_empty_mask_does_not_crash(self) -> None:
        model = build_pointnet_classifier(
            num_points=128, mlp_channels=(16, 32), global_dim=32, head_hidden_dim=16,
            dropout=0.0,
        )
        model.eval()
        inputs = torch.zeros(2, 1, 16, 16, 16)
        with torch.no_grad():
            logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (2,))
        self.assertTrue(torch.isfinite(logits).all())

    def test_small_foreground_is_oversampled(self) -> None:
        # Create a volume with very few foreground voxels (< num_points).
        volume = torch.zeros(1, 1, 16, 16, 16)
        volume[0, 0, 4:6, 4:6, 4:6] = 1.0  # 8 foreground voxels
        points = volume_to_pointcloud(volume, num_points=64, training=False)
        self.assertEqual(tuple(points.shape), (1, 3, 64))
        # All points must be in the normalized valid range.
        self.assertTrue(torch.isfinite(points).all())
        self.assertTrue((points.abs() <= 1.0 + 1e-6).all())

    def test_point_features_four_accepts_occupancy(self) -> None:
        model = build_pointnet_classifier(
            num_points=128, point_features=4, mlp_channels=(16, 32), global_dim=32,
            head_hidden_dim=0, dropout=0.0,
        )
        inputs = self._make_binary_volume(1, 16)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_input_transform_runs(self) -> None:
        model = build_pointnet_classifier(
            num_points=128, mlp_channels=(16, 32), global_dim=32, head_hidden_dim=16,
            dropout=0.0, use_input_transform=True,
        )
        inputs = self._make_binary_volume(1, 16)
        logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_deterministic_sampling_in_eval(self) -> None:
        model = build_pointnet_classifier(
            num_points=128, mlp_channels=(16, 32), global_dim=32, head_hidden_dim=16,
            dropout=0.0,
        )
        model.eval()
        volume = self._make_binary_volume(1, 16, density=0.5)
        with torch.no_grad():
            a = model(volume)
            b = model(volume)
        self.assertTrue(torch.equal(a, b))


class FactoryTests(unittest.TestCase):
    def test_factory_builds_resnet_by_default(self) -> None:
        cfg = ModelConfig(base_channels=8, dropout=0.0, use_tabular_features=False)
        model = build_model(cfg, num_tabular_features=0)
        inputs = torch.randn(1, 1, 32, 32, 32)
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_factory_builds_unet(self) -> None:
        cfg = ModelConfig(
            architecture="unet3d", dropout=0.0, use_tabular_features=False,
            unet_depth=3, unet_base_channels=8, unet_channel_multiplier=2,
        )
        model = build_model(cfg, num_tabular_features=0)
        inputs = torch.randn(1, 1, 32, 32, 32)
        model.eval()
        with torch.no_grad():
            logits = model(inputs)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_factory_builds_pointnet(self) -> None:
        cfg = ModelConfig(
            architecture="pointnet", dropout=0.0, use_tabular_features=False,
            pointnet_num_points=256, pointnet_mlp_channels=(32, 64),
            pointnet_global_dim=64, pointnet_head_hidden_dim=32,
        )
        model = build_model(cfg, num_tabular_features=0)
        volume = (torch.rand(2, 1, 16, 16, 16) > 0.7).float()
        model.eval()
        with torch.no_grad():
            logits = model(volume)
        self.assertEqual(tuple(logits.shape), (2,))

    def test_factory_rejects_unknown_architecture(self) -> None:
        cfg = ModelConfig(architecture="bogus")
        with self.assertRaises(ValueError):
            build_model(cfg, num_tabular_features=0)


if __name__ == "__main__":
    unittest.main()
