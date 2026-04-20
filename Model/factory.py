from __future__ import annotations

from torch import nn

from Model.pointnet import build_pointnet_classifier
from Model.resnet3d import build_resnet3d
from Model.unet3d import build_unet3d_classifier
from Utils.config import ModelConfig


SUPPORTED_ARCHITECTURES = ("resnet3d", "unet3d", "pointnet")


def build_model(model_config: ModelConfig, num_tabular_features: int = 0) -> nn.Module:
    """Construct a 3D classification backbone from ModelConfig.

    Centralising model instantiation here keeps engine.py / ensemble.py agnostic
    to architecture-specific arguments and lets HPO sample ``architecture``
    without touching training code.
    """
    architecture = getattr(model_config, "architecture", "resnet3d") or "resnet3d"
    architecture = architecture.lower()

    if architecture == "resnet3d":
        return build_resnet3d(
            depth=model_config.depth,
            in_channels=model_config.in_channels,
            base_channels=model_config.base_channels,
            dropout=model_config.dropout,
            num_classes=model_config.num_classes,
            num_tabular_features=num_tabular_features,
            tabular_hidden_dim=model_config.tabular_hidden_dim,
            norm_type=getattr(model_config, "norm_type", "batch"),
            group_norm_groups=getattr(model_config, "group_norm_groups", 8),
        )
    if architecture == "pointnet":
        return build_pointnet_classifier(
            num_points=getattr(model_config, "pointnet_num_points", 1024),
            point_features=getattr(model_config, "pointnet_point_features", 3),
            mlp_channels=tuple(getattr(model_config, "pointnet_mlp_channels", (64, 128, 256))),
            global_dim=getattr(model_config, "pointnet_global_dim", 512),
            head_hidden_dim=getattr(model_config, "pointnet_head_hidden_dim", 128),
            dropout=model_config.dropout,
            num_classes=model_config.num_classes,
            num_tabular_features=num_tabular_features,
            tabular_hidden_dim=model_config.tabular_hidden_dim,
            norm_type=getattr(model_config, "norm_type", "batch"),
            group_norm_groups=getattr(model_config, "group_norm_groups", 8),
            binary_threshold=getattr(model_config, "pointnet_binary_threshold", 0.5),
            use_input_transform=getattr(model_config, "pointnet_use_input_transform", False),
        )
    if architecture == "unet3d":
        bottleneck = getattr(model_config, "unet_bottleneck_channels", None)
        return build_unet3d_classifier(
            in_channels=model_config.in_channels,
            base_channels=getattr(model_config, "unet_base_channels", 16),
            depth=getattr(model_config, "unet_depth", 4),
            channel_multiplier=getattr(model_config, "unet_channel_multiplier", 2),
            bottleneck_channels=bottleneck,
            dropout=model_config.dropout,
            num_classes=model_config.num_classes,
            num_tabular_features=num_tabular_features,
            tabular_hidden_dim=model_config.tabular_hidden_dim,
            norm_type=getattr(model_config, "norm_type", "batch"),
            group_norm_groups=getattr(model_config, "group_norm_groups", 8),
        )
    raise ValueError(
        f"Unsupported architecture: {architecture!r}. "
        f"Expected one of {SUPPORTED_ARCHITECTURES}."
    )
