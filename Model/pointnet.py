from __future__ import annotations

import torch
from torch import nn


def _make_norm_1d(channels: int, norm_type: str, group_norm_groups: int) -> nn.Module:
    if norm_type == "batch":
        return nn.BatchNorm1d(channels)
    if norm_type == "group":
        groups = max(1, min(int(group_norm_groups), channels))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(num_groups=groups, num_channels=channels)
    raise ValueError(f"Unsupported norm_type: {norm_type}")


def volume_to_pointcloud(
    volume: torch.Tensor,
    num_points: int,
    training: bool,
    binary_threshold: float = 0.5,
) -> torch.Tensor:
    """Convert a batch of binary 3D volumes to normalized point clouds.

    Empty masks return zero points (a valid fallback that keeps downstream
    shared-MLP / max-pool math well-defined).
    """
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError(
            f"volume_to_pointcloud expects shape (B, 1, D, H, W); got {tuple(volume.shape)}."
        )
    if num_points < 1:
        raise ValueError(f"num_points must be >= 1, got {num_points}.")

    B, _, D, H, W = volume.shape
    device = volume.device
    dtype = volume.dtype if volume.is_floating_point() else torch.float32
    scale = torch.tensor(
        [max(D - 1, 1), max(H - 1, 1), max(W - 1, 1)],
        device=device,
        dtype=dtype,
    )

    out = torch.zeros(B, 3, num_points, device=device, dtype=dtype)
    for b in range(B):
        mask = volume[b, 0] > binary_threshold
        coords = mask.nonzero(as_tuple=False)
        n = int(coords.shape[0])
        if n == 0:
            continue
        coords_f = coords.to(dtype)
        normalized = (coords_f / scale) * 2.0 - 1.0  # (n, 3) in [-1, 1]

        if training:
            idx = torch.randint(0, n, (num_points,), device=device)
        elif n >= num_points:
            idx = torch.linspace(0, n - 1, steps=num_points, device=device).long()
        else:
            idx = torch.arange(num_points, device=device) % n

        out[b] = normalized.index_select(0, idx).t().contiguous()
    return out


class _InputTransform(nn.Module):
    """Tiny T-Net predicting a per-sample 3x3 matrix, initialized to identity."""

    def __init__(
        self,
        in_dim: int = 3,
        mlp_channels: tuple[int, ...] = (64, 128),
        hidden_dim: int = 64,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        layers: list[nn.Module] = []
        prev = in_dim
        for channels in mlp_channels:
            layers.extend([
                nn.Conv1d(prev, channels, kernel_size=1, bias=False),
                _make_norm_1d(channels, norm_type, group_norm_groups),
                nn.ReLU(inplace=True),
            ])
            prev = channels
        self.shared_mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.Linear(prev, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, in_dim * in_dim),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("identity", torch.eye(in_dim).flatten(), persistent=False)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        features = self.shared_mlp(points)
        pooled = features.max(dim=2).values
        matrix = self.head(pooled) + self.identity
        matrix = matrix.view(-1, self.in_dim, self.in_dim)
        return torch.bmm(matrix, points)


class PointNetClassifier(nn.Module):
    """Compact PointNet-style binary classifier operating on 3D binary masks.

    The volume input is converted on-the-fly to a fixed-size point cloud of
    foreground voxels so the training engine can keep feeding 3D volumes.
    """

    def __init__(
        self,
        num_points: int = 1024,
        point_features: int = 3,
        mlp_channels: tuple[int, ...] = (64, 128, 256),
        global_dim: int = 512,
        head_hidden_dim: int = 128,
        dropout: float = 0.3,
        num_classes: int = 1,
        num_tabular_features: int = 0,
        tabular_hidden_dim: int = 16,
        norm_type: str = "batch",
        group_norm_groups: int = 8,
        binary_threshold: float = 0.5,
        use_input_transform: bool = False,
    ) -> None:
        super().__init__()
        if num_points < 1:
            raise ValueError(f"num_points must be >= 1, got {num_points}.")
        if point_features not in (3, 4):
            raise ValueError(
                f"point_features must be 3 (xyz) or 4 (xyz+occupancy), got {point_features}."
            )
        if global_dim < 1:
            raise ValueError(f"global_dim must be positive, got {global_dim}.")
        if not mlp_channels:
            raise ValueError("mlp_channels must contain at least one layer.")
        if dropout < 0.0 or dropout >= 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}.")
        if num_classes < 1:
            raise ValueError(f"num_classes must be >= 1, got {num_classes}.")

        self.num_points = int(num_points)
        self.point_features = int(point_features)
        self.binary_threshold = float(binary_threshold)
        self.num_tabular_features = int(num_tabular_features)
        self.use_input_transform = bool(use_input_transform)

        if self.use_input_transform:
            self.input_transform: nn.Module | None = _InputTransform(
                in_dim=3,
                mlp_channels=(64, 128),
                hidden_dim=64,
                norm_type=norm_type,
                group_norm_groups=group_norm_groups,
            )
        else:
            self.input_transform = None

        shared_layers: list[nn.Module] = []
        prev = self.point_features
        for channels in mlp_channels:
            shared_layers.extend([
                nn.Conv1d(prev, channels, kernel_size=1, bias=False),
                _make_norm_1d(channels, norm_type, group_norm_groups),
                nn.ReLU(inplace=True),
            ])
            prev = channels
        shared_layers.extend([
            nn.Conv1d(prev, int(global_dim), kernel_size=1, bias=False),
            _make_norm_1d(int(global_dim), norm_type, group_norm_groups),
            nn.ReLU(inplace=True),
        ])
        self.shared_mlp = nn.Sequential(*shared_layers)

        self.dropout = nn.Dropout(float(dropout))
        classifier_in = int(global_dim)

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
            classifier_in += tabular_hidden_dim
        else:
            self.tabular_head = None

        head_hidden_dim = int(head_hidden_dim)
        if head_hidden_dim > 0:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_in, head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(float(dropout)),
                nn.Linear(head_hidden_dim, num_classes),
            )
        else:
            self.classifier = nn.Linear(classifier_in, num_classes)

        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.GroupNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.abs().sum() == 0:
                    continue  # preserve identity-init T-Net head
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _prepare_points(self, volume: torch.Tensor) -> torch.Tensor:
        points = volume_to_pointcloud(
            volume,
            num_points=self.num_points,
            training=self.training,
            binary_threshold=self.binary_threshold,
        )
        if self.point_features == 4:
            # Append occupancy feature (=1 for all points, including fallback zeros).
            occupancy = torch.ones(
                points.shape[0], 1, points.shape[2], device=points.device, dtype=points.dtype
            )
            points = torch.cat([points, occupancy], dim=1)
        return points

    def forward(
        self,
        inputs: torch.Tensor,
        tabular_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        points = self._prepare_points(inputs)
        if self.input_transform is not None:
            # Only transform the xyz channels; leave extras (occupancy) untouched.
            xyz = points[:, :3]
            rest = points[:, 3:]
            xyz = self.input_transform(xyz)
            points = torch.cat([xyz, rest], dim=1) if rest.shape[1] > 0 else xyz

        features = self.shared_mlp(points)
        global_feat = features.max(dim=2).values
        outputs = self.dropout(global_feat)

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

        logits = self.classifier(outputs).squeeze(-1)
        return logits


def build_pointnet_classifier(
    num_points: int = 1024,
    point_features: int = 3,
    mlp_channels: tuple[int, ...] = (64, 128, 256),
    global_dim: int = 512,
    head_hidden_dim: int = 128,
    dropout: float = 0.3,
    num_classes: int = 1,
    num_tabular_features: int = 0,
    tabular_hidden_dim: int = 16,
    norm_type: str = "batch",
    group_norm_groups: int = 8,
    binary_threshold: float = 0.5,
    use_input_transform: bool = False,
) -> PointNetClassifier:
    return PointNetClassifier(
        num_points=num_points,
        point_features=point_features,
        mlp_channels=tuple(mlp_channels),
        global_dim=global_dim,
        head_hidden_dim=head_hidden_dim,
        dropout=dropout,
        num_classes=num_classes,
        num_tabular_features=num_tabular_features,
        tabular_hidden_dim=tabular_hidden_dim,
        norm_type=norm_type,
        group_norm_groups=group_norm_groups,
        binary_threshold=binary_threshold,
        use_input_transform=use_input_transform,
    )
