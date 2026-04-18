from __future__ import annotations

import torch
from torch import nn


def conv3x3x3(in_channels: int, out_channels: int, stride: int = 1) -> nn.Conv3d:
    return nn.Conv3d(
        in_channels,
        out_channels,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(in_channels, out_channels, stride=stride)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(inputs)

        outputs = self.conv1(inputs)
        outputs = self.bn1(outputs)
        outputs = self.relu1(outputs)
        outputs = self.conv2(outputs)
        outputs = self.bn2(outputs)
        outputs = outputs + residual
        outputs = self.relu2(outputs)
        return outputs


class ResNet3D(nn.Module):
    def __init__(
        self,
        layers: tuple[int, int, int, int],
        in_channels: int = 1,
        base_channels: int = 32,
        dropout: float = 0.2,
        num_classes: int = 1,
        num_tabular_features: int = 0,
        tabular_hidden_dim: int = 16,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.num_tabular_features = int(num_tabular_features)

        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
        )

        self.layer1 = self._make_layer(base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout)

        classifier_channels = base_channels * 8
        if self.num_tabular_features > 0:
            tabular_hidden_dim = int(tabular_hidden_dim)
            if tabular_hidden_dim <= 0:
                raise ValueError("tabular_hidden_dim must be positive when tabular features are enabled.")
            self.tabular_head = nn.Sequential(
                nn.Linear(self.num_tabular_features, tabular_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
            classifier_channels += tabular_hidden_dim
        else:
            self.tabular_head = None

        self.fc = nn.Linear(classifier_channels, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _make_layer(self, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock3D(self.in_channels, out_channels, stride=stride)]
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock3D(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor, tabular_features: torch.Tensor | None = None) -> torch.Tensor:
        outputs = self.stem(inputs)
        outputs = self.layer1(outputs)
        outputs = self.layer2(outputs)
        outputs = self.layer3(outputs)
        outputs = self.layer4(outputs)
        outputs = self.avgpool(outputs).flatten(1)
        outputs = self.dropout(outputs)

        if self.tabular_head is not None:
            if tabular_features is None:
                raise ValueError("tabular_features must be provided when num_tabular_features > 0.")
            if tabular_features.ndim == 1:
                tabular_features = tabular_features.unsqueeze(0)
            if tabular_features.shape[1] != self.num_tabular_features:
                raise ValueError(
                    f"Expected {self.num_tabular_features} tabular features, "
                    f"got {tabular_features.shape[1]}."
                )
            tabular_features = tabular_features.to(device=outputs.device, dtype=outputs.dtype)
            outputs = torch.cat([outputs, self.tabular_head(tabular_features)], dim=1)

        logits = self.fc(outputs).squeeze(-1)
        return logits


def build_resnet3d(
    depth: int = 18,
    in_channels: int = 1,
    base_channels: int = 32,
    dropout: float = 0.2,
    num_classes: int = 1,
    num_tabular_features: int = 0,
    tabular_hidden_dim: int = 16,
) -> ResNet3D:
    variants = {
        18: (2, 2, 2, 2),
        34: (3, 4, 6, 3),
    }
    if depth not in variants:
        raise ValueError(f"Unsupported ResNet3D depth: {depth}. Supported depths: {sorted(variants)}")
    return ResNet3D(
        layers=variants[depth],
        in_channels=in_channels,
        base_channels=base_channels,
        dropout=dropout,
        num_classes=num_classes,
        num_tabular_features=num_tabular_features,
        tabular_hidden_dim=tabular_hidden_dim,
    )
