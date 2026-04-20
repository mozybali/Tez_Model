from __future__ import annotations

import torch
from torch import nn


def _make_norm(channels: int, norm_type: str, group_norm_groups: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm3d(channels)
    if norm_type == "group":
        groups = max(1, min(int(group_norm_groups), channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


class _ConvBlock3D(nn.Module):
    """Two 3x3x3 convs with normalization and ReLU — the U-Net workhorse block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type, group_norm_groups),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm_type, group_norm_groups),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class UNet3DClassifier(nn.Module):
    """Lightweight 3D U-Net adapted for binary classification.

    The encoder/decoder structure produces multi-scale features; we pool the
    decoder's top feature map globally and feed it to a small classifier head
    instead of a segmentation 1x1 conv.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        depth: int = 4,
        channel_multiplier: int = 2,
        bottleneck_channels: int | None = None,
        dropout: float = 0.2,
        num_classes: int = 1,
        num_tabular_features: int = 0,
        tabular_hidden_dim: int = 16,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        if depth < 2:
            raise ValueError(f"UNet3D depth must be >= 2, got {depth}.")
        if base_channels < 1:
            raise ValueError(f"UNet3D base_channels must be positive, got {base_channels}.")
        if channel_multiplier < 1:
            raise ValueError(
                f"UNet3D channel_multiplier must be >= 1, got {channel_multiplier}."
            )
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"UNet3D dropout must be in [0, 1), got {dropout}.")
        if num_classes < 1:
            raise ValueError(f"UNet3D num_classes must be >= 1, got {num_classes}.")

        self.depth = int(depth)
        self.num_tabular_features = int(num_tabular_features)
        self.norm_type = norm_type
        self.group_norm_groups = int(group_norm_groups)

        encoder_channels: list[int] = []
        self.encoder_blocks = nn.ModuleList()
        self.pool_layers = nn.ModuleList()

        prev_channels = in_channels
        for level in range(self.depth):
            level_channels = base_channels * (channel_multiplier ** level)
            self.encoder_blocks.append(
                _ConvBlock3D(prev_channels, level_channels, norm_type, group_norm_groups)
            )
            encoder_channels.append(level_channels)
            if level < self.depth - 1:
                self.pool_layers.append(nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True))
            prev_channels = level_channels

        bottleneck_out = (
            int(bottleneck_channels)
            if bottleneck_channels is not None
            else encoder_channels[-1] * channel_multiplier
        )
        if bottleneck_out < 1:
            raise ValueError(
                f"UNet3D bottleneck_channels must be positive, got {bottleneck_out}."
            )
        self.bottleneck_pool = nn.MaxPool3d(kernel_size=2, stride=2, ceil_mode=True)
        self.bottleneck = _ConvBlock3D(
            encoder_channels[-1], bottleneck_out, norm_type, group_norm_groups
        )

        self.up_layers = nn.ModuleList()
        self.decoder_blocks = nn.ModuleList()
        prev_channels = bottleneck_out
        for level in reversed(range(self.depth)):
            skip_channels = encoder_channels[level]
            self.up_layers.append(
                nn.ConvTranspose3d(prev_channels, skip_channels, kernel_size=2, stride=2)
            )
            self.decoder_blocks.append(
                _ConvBlock3D(
                    skip_channels + skip_channels, skip_channels, norm_type, group_norm_groups
                )
            )
            prev_channels = skip_channels

        classifier_channels = prev_channels
        self.global_pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(float(dropout))

        if self.num_tabular_features > 0:
            tabular_hidden_dim = int(tabular_hidden_dim)
            if tabular_hidden_dim <= 0:
                raise ValueError(
                    "tabular_hidden_dim must be positive when tabular features are enabled."
                )
            self.tabular_head = nn.Sequential(
                nn.Linear(self.num_tabular_features, tabular_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
            )
            classifier_channels += tabular_hidden_dim
        else:
            self.tabular_head = None

        self.fc = nn.Linear(classifier_channels, num_classes)

        for module in self.modules():
            if isinstance(module, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm3d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    @staticmethod
    def _match_spatial(up: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Crop/center the upsampled tensor to match the skip connection's spatial dims.

        Needed because ceil_mode pooling on odd sizes plus ConvTranspose3d can
        produce off-by-one mismatches at decoder levels.
        """
        if up.shape[2:] == skip.shape[2:]:
            return up
        diffs = [skip.size(dim) - up.size(dim) for dim in (2, 3, 4)]
        pad = []
        for diff in reversed(diffs):
            pad.append(diff // 2)
            pad.append(diff - diff // 2)
        if any(p != 0 for p in pad):
            up = torch.nn.functional.pad(up, pad)
        if up.shape[2:] != skip.shape[2:]:
            up = up[
                :, :,
                : skip.size(2),
                : skip.size(3),
                : skip.size(4),
            ]
        return up

    def forward(
        self,
        inputs: torch.Tensor,
        tabular_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        skips: list[torch.Tensor] = []
        outputs = inputs
        for level, block in enumerate(self.encoder_blocks):
            outputs = block(outputs)
            skips.append(outputs)
            if level < self.depth - 1:
                outputs = self.pool_layers[level](outputs)

        outputs = self.bottleneck_pool(outputs)
        outputs = self.bottleneck(outputs)

        for up, block, skip in zip(self.up_layers, self.decoder_blocks, reversed(skips)):
            outputs = up(outputs)
            outputs = self._match_spatial(outputs, skip)
            outputs = torch.cat([outputs, skip], dim=1)
            outputs = block(outputs)

        outputs = self.global_pool(outputs).flatten(1)
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


def build_unet3d_classifier(
    in_channels: int = 1,
    base_channels: int = 16,
    depth: int = 4,
    channel_multiplier: int = 2,
    bottleneck_channels: int | None = None,
    dropout: float = 0.2,
    num_classes: int = 1,
    num_tabular_features: int = 0,
    tabular_hidden_dim: int = 16,
    norm_type: str = "batch",
    group_norm_groups: int = 8,
) -> UNet3DClassifier:
    return UNet3DClassifier(
        in_channels=in_channels,
        base_channels=base_channels,
        depth=depth,
        channel_multiplier=channel_multiplier,
        bottleneck_channels=bottleneck_channels,
        dropout=dropout,
        num_classes=num_classes,
        num_tabular_features=num_tabular_features,
        tabular_hidden_dim=tabular_hidden_dim,
        norm_type=norm_type,
        group_norm_groups=group_norm_groups,
    )
