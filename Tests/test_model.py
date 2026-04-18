from __future__ import annotations

import unittest

import torch

from Model.resnet3d import build_resnet3d


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


if __name__ == "__main__":
    unittest.main()
