"""Produce a minimal V2 run artifact bundle so the bridge can run end-to-end.

Builds a default ResNet3D, computes tabular feature stats from ALAN/metadata.csv,
and writes:
    Classification/outputs/predictor_v2/best_model.pt
    Classification/outputs/predictor_v2/config.json

Note: weights are randomly initialised — replace with a real training run when
available. The bridge will load whatever checkpoint exists; this only wires up
the artifact layout the bridge expects.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from Model.factory import build_model  # noqa: E402
from Utils.config import (  # noqa: E402
    AugmentationConfig,
    DataConfig,
    ModelConfig,
    TrainConfig,
    to_serializable,
)


TABULAR_FEATURE_NAMES = ("log_voxel_count_z", "side_is_left")


def _tabular_stats_from_metadata(metadata_csv: Path) -> dict[str, float]:
    voxels: list[float] = []
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                voxels.append(float(row["voxel_count"]))
            except (KeyError, ValueError):
                continue
    if not voxels:
        return {"log_voxel_count_mean": 0.0, "log_voxel_count_std": 1.0}
    log_voxels = [math.log1p(max(value, 0.0)) for value in voxels]
    mean = sum(log_voxels) / len(log_voxels)
    var = sum((value - mean) ** 2 for value in log_voxels) / len(log_voxels)
    std = math.sqrt(var)
    if std < 1e-8:
        std = 1.0
    return {"log_voxel_count_mean": mean, "log_voxel_count_std": std}


def main() -> int:
    output_dir = ROOT / "outputs" / "predictor_v2"
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = DataConfig().resolved()
    augmentation_config = AugmentationConfig()
    model_config = ModelConfig()
    train_config = TrainConfig()

    metadata_csv = ROOT / "ALAN" / "metadata.csv"
    if model_config.use_tabular_features and metadata_csv.exists():
        stats = _tabular_stats_from_metadata(metadata_csv)
    else:
        stats = None

    num_tabular = len(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else 0
    model = build_model(model_config=model_config, num_tabular_features=num_tabular)
    model.eval()

    checkpoint = {
        "epoch": 0,
        "model_state_dict": model.state_dict(),
        "score": 0.0,
        "note": "random-init bootstrap; replace with a real training run.",
    }
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)

    config_payload = {
        "data": to_serializable(data_config),
        "augmentation": to_serializable(augmentation_config),
        "model": to_serializable(model_config),
        "train": to_serializable(train_config),
        "positive_class_weight": 1.0,
        "optimal_threshold": 0.5,
        "fixed_threshold": 0.5,
        "temperature": 1.0,
        "tabular_feature_names": list(TABULAR_FEATURE_NAMES) if model_config.use_tabular_features else [],
        "tabular_feature_stats": stats,
        "test_eval_error": None,
    }
    config_path = output_dir / "config.json"
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    print(f"wrote {checkpoint_path}", file=sys.stderr)
    print(f"wrote {config_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
