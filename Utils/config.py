from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any


_VALID_NAN_STRATEGIES = frozenset(
    {"none", "drop_record", "fill_zero", "fill_mean", "fill_median", "fill_constant"}
)


def _to_tuple3(values: tuple[int, int, int] | list[int] | int) -> tuple[int, int, int]:
    if isinstance(values, int):
        return (values, values, values)
    if len(values) != 3:
        raise ValueError("target_shape must contain exactly three integers.")
    return (int(values[0]), int(values[1]), int(values[2]))


@dataclass(slots=True)
class DataConfig:
    info_csv: Path = Path("ALAN/info.csv")
    volumes_dir: Path = Path("ALAN/alan")
    metadata_csv: Path = Path("ALAN/metadata.csv")
    summary_json: Path = Path("ALAN/summary.json")
    train_subset: str = "ZS-train"
    val_subset: str = "ZS-dev"
    test_subset: str = "ZS-test"
    target_shape: tuple[int, int, int] = (64, 64, 64)
    use_bbox_crop: bool = True
    bbox_margin: int = 8
    pad_to_cube: bool = True
    canonicalize_right: bool = False
    right_flip_axis: int = 0
    nan_strategy: str = "none"
    nan_fill_value: float = 0.0

    def __post_init__(self) -> None:
        if self.nan_strategy not in _VALID_NAN_STRATEGIES:
            raise ValueError(
                f"Invalid nan_strategy '{self.nan_strategy}'. "
                f"Must be one of {sorted(_VALID_NAN_STRATEGIES)}"
            )

    def resolved(self) -> "DataConfig":
        return DataConfig(
            info_csv=self.info_csv,
            volumes_dir=self.volumes_dir,
            metadata_csv=self.metadata_csv,
            summary_json=self.summary_json,
            train_subset=self.train_subset,
            val_subset=self.val_subset,
            test_subset=self.test_subset,
            target_shape=_to_tuple3(self.target_shape),
            use_bbox_crop=self.use_bbox_crop,
            bbox_margin=int(self.bbox_margin),
            pad_to_cube=self.pad_to_cube,
            canonicalize_right=self.canonicalize_right,
            right_flip_axis=int(self.right_flip_axis),
            nan_strategy=self.nan_strategy,
            nan_fill_value=float(self.nan_fill_value),
        )


@dataclass(slots=True)
class AugmentationConfig:
    enabled: bool = True
    flip_probability: float = 0.5
    flip_axes: tuple[int, ...] = (1, 2)
    affine_probability: float = 0.6
    rotation_degrees: float = 10.0
    translation_fraction: float = 0.05
    scale_min: float = 0.9
    scale_max: float = 1.1
    morphology_probability: float = 0.1


@dataclass(slots=True)
class ModelConfig:
    architecture: str = "resnet3d"  # "resnet3d", "unet3d" or "pointnet"
    depth: int = 18
    in_channels: int = 1
    base_channels: int = 32
    dropout: float = 0.3
    num_classes: int = 1
    use_tabular_features: bool = True
    tabular_hidden_dim: int = 16
    norm_type: str = "batch"  # "batch" or "group" — group helps when batch_size is small.
    group_norm_groups: int = 8
    # U-Net–specific fields (ignored when architecture == "resnet3d")
    unet_depth: int = 4
    unet_base_channels: int = 16
    unet_channel_multiplier: int = 2
    unet_bottleneck_channels: int | None = None
    # PointNet-specific fields (ignored when architecture != "pointnet")
    pointnet_num_points: int = 1024
    pointnet_point_features: int = 3
    pointnet_mlp_channels: tuple[int, ...] = (64, 128, 256)
    pointnet_global_dim: int = 512
    pointnet_head_hidden_dim: int = 128
    pointnet_binary_threshold: float = 0.5
    pointnet_use_input_transform: bool = False


@dataclass(slots=True)
class TrainConfig:
    output_dir: Path = Path("outputs/baseline")
    epochs: int = 20
    batch_size: int = 8
    num_workers: int = 0
    pin_memory: bool = True
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    optimizer_name: str = "adamw"
    sgd_momentum: float = 0.9
    scheduler_name: str = "cosine"
    device: str = "auto"
    seed: int = 42
    early_stopping_patience: int = 10
    gradient_clip_norm: float | None = 1.0
    decision_threshold: float = 0.5
    primary_metric: str = "pr_auc"
    amp: bool = True
    warmup_epochs: int = 3
    pos_weight_strategy: str = "inverse"
    loss_type: str = "bce"
    focal_gamma: float = 2.0
    cv_folds: int = 0
    use_weighted_sampler: bool = False
    calibrate_temperature: bool = True
    threshold_selection: str = "youden"  # "youden", "f1", "fbeta", or "fixed"
    threshold_fbeta: float = 1.0  # used when threshold_selection == "fbeta"; >1 favors recall
    calibration_method: str = "temperature"  # "temperature", "isotonic", or "temperature+isotonic"
    tta_enabled: bool = False  # flip-based test-time augmentation during collect_predictions


@dataclass(slots=True)
class SearchConfig:
    study_name: str = "alan_resnet3d"
    output_dir: Path = Path("outputs/optuna")
    n_trials: int = 15
    timeout_seconds: int | None = None
    sampler_seed: int = 42


def to_serializable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: to_serializable(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_serializable(item) for item in value]
    return value
