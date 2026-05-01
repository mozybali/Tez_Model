from __future__ import annotations

import csv
import hashlib
import inspect
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as torch_functional
from torch.utils.data import Dataset

from Preprocessing.analyze_dataset import ensure_metadata


_VALID_CACHE_MODES = {"none", "memory", "disk"}
_PREPROCESS_CACHE_VERSION = 2


@dataclass(frozen=True, slots=True)
class AlanRecord:
    roi_id: str
    subset: str
    label: int
    side: str
    volume_path: Path
    voxel_count: int
    bbox_min: tuple[int, int, int]
    bbox_max: tuple[int, int, int]
    nan_count: int = 0
    nan_ratio: float = 0.0
    has_nan: bool = False


def load_records(
    info_csv: Path,
    volumes_dir: Path,
    metadata_csv: Path,
    summary_json: Path | None = None,
) -> list[AlanRecord]:
    metadata_path = ensure_metadata(
        info_csv=info_csv,
        volumes_dir=volumes_dir,
        metadata_csv=metadata_csv,
        summary_json=summary_json,
    )
    records: list[AlanRecord] = []
    with metadata_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            volume_entry = Path(row["volume_path"])
            if volume_entry.is_absolute() or len(volume_entry.parts) > 1:
                volume_path = volume_entry
            else:
                volume_path = volumes_dir / volume_entry
            records.append(
                AlanRecord(
                    roi_id=row["ROI_id"],
                    subset=row["subset"],
                    label=int(row["label_int"]),
                    side=row["side"],
                    volume_path=volume_path,
                    voxel_count=int(row["voxel_count"]),
                    bbox_min=(int(row["bbox_min_d"]), int(row["bbox_min_h"]), int(row["bbox_min_w"])),
                    bbox_max=(int(row["bbox_max_d"]), int(row["bbox_max_h"]), int(row["bbox_max_w"])),
                    nan_count=int(row.get("nan_count", 0)),
                    nan_ratio=float(row.get("nan_ratio", 0.0)),
                    has_nan=bool(int(row.get("has_nan", 0))),
                )
            )
    return records


def split_records(
    records: list[AlanRecord],
    train_subset: str = "ZS-train",
    val_subset: str = "ZS-dev",
    test_subset: str = "ZS-test",
) -> dict[str, list[AlanRecord]]:
    return {
        "train": [record for record in records if record.subset == train_subset],
        "val": [record for record in records if record.subset == val_subset],
        "test": [record for record in records if record.subset == test_subset],
    }


def apply_nan_strategy(volume: np.ndarray, strategy: str, fill_value: float = 0.0) -> np.ndarray:
    if strategy == "none":
        return volume
    if strategy == "fill_zero":
        return np.nan_to_num(volume, nan=0.0)
    if strategy == "fill_constant":
        return np.nan_to_num(volume, nan=fill_value)
    if strategy == "fill_mean":
        valid = volume[~np.isnan(volume)]
        replacement = float(valid.mean()) if valid.size > 0 else 0.0
        return np.nan_to_num(volume, nan=replacement)
    if strategy == "fill_median":
        valid = volume[~np.isnan(volume)]
        replacement = float(np.median(valid)) if valid.size > 0 else 0.0
        return np.nan_to_num(volume, nan=replacement)
    return volume


def crop_to_bbox(volume: np.ndarray, bbox_min: tuple[int, int, int], bbox_max: tuple[int, int, int], margin: int) -> np.ndarray:
    crop_slices = []
    for axis, size in enumerate(volume.shape):
        start = max(0, bbox_min[axis] - margin)
        stop = min(size, bbox_max[axis] + margin + 1)
        crop_slices.append(slice(start, stop))
    return volume[tuple(crop_slices)]


def pad_to_cube(volume: np.ndarray) -> np.ndarray:
    max_dim = int(max(volume.shape))
    padding = []
    for current in volume.shape:
        total = max_dim - current
        left = total // 2
        right = total - left
        padding.append((left, right))
    return np.pad(volume, pad_width=padding, mode="constant", constant_values=0)


def resize_volume(volume: np.ndarray, target_shape: tuple[int, int, int]) -> torch.Tensor:
    tensor = torch.from_numpy(volume.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = torch_functional.interpolate(
        tensor,
        size=target_shape,
        mode="trilinear",
        align_corners=False,
    )
    return resized.squeeze(0)


def infer_positive_class_weight(records: list[AlanRecord]) -> float:
    positives = sum(record.label for record in records)
    negatives = len(records) - positives
    if positives == 0:
        return 1.0
    return negatives / positives


class AlanKidneyDataset(Dataset):
    def __init__(
        self,
        records: list[AlanRecord],
        target_shape: tuple[int, int, int] = (64, 64, 64),
        use_bbox_crop: bool = True,
        bbox_margin: int = 8,
        pad_to_cube_input: bool = True,
        canonicalize_right: bool = False,
        right_flip_axis: int = 0,
        nan_strategy: str = "none",
        nan_fill_value: float = 0.0,
        cache_mode: str = "none",
        cache_dir: Path | None = None,
        transform=None,
    ) -> None:
        self.records = records
        self.target_shape = tuple(int(value) for value in target_shape)
        self.use_bbox_crop = use_bbox_crop
        self.bbox_margin = int(bbox_margin)
        self.pad_to_cube_input = pad_to_cube_input
        self.canonicalize_right = canonicalize_right
        self.right_flip_axis = int(right_flip_axis)
        self.nan_strategy = nan_strategy
        self.nan_fill_value = nan_fill_value
        if cache_mode not in _VALID_CACHE_MODES:
            raise ValueError(
                f"Invalid cache_mode '{cache_mode}'. Must be one of {sorted(_VALID_CACHE_MODES)}."
            )
        self.cache_mode = cache_mode
        self.cache_dir = Path(cache_dir) if cache_dir is not None else Path("outputs/preprocessed_cache")
        self._memory_cache: dict[str, torch.Tensor] = {}
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def _cache_key(self, record: AlanRecord) -> str:
        try:
            stat = record.volume_path.stat()
            volume_signature = {
                "path": str(record.volume_path.resolve()),
                "mtime_ns": stat.st_mtime_ns,
                "size": stat.st_size,
            }
        except FileNotFoundError:
            volume_signature = {"path": str(record.volume_path), "missing": True}

        payload = {
            "version": _PREPROCESS_CACHE_VERSION,
            "preprocess_fingerprint": _preprocess_cache_fingerprint(),
            "roi_id": record.roi_id,
            "volume": volume_signature,
            "target_shape": self.target_shape,
            "use_bbox_crop": self.use_bbox_crop,
            "bbox_margin": self.bbox_margin,
            "pad_to_cube_input": self.pad_to_cube_input,
            "canonicalize_right": self.canonicalize_right,
            "right_flip_axis": self.right_flip_axis,
            "nan_strategy": self.nan_strategy,
            "nan_fill_value": self.nan_fill_value,
            "bbox_min": record.bbox_min,
            "bbox_max": record.bbox_max,
            "side": record.side,
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha1(raw).hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def _load_disk_cache(self, key: str) -> torch.Tensor | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        array = np.load(path)
        return torch.from_numpy(array.astype(np.float32, copy=False))

    def _save_disk_cache(self, key: str, tensor: torch.Tensor) -> None:
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(key)
        if path.exists():
            return
        tmp_path = path.with_name(f"{path.stem}.{os.getpid()}.{id(self)}.tmp.npy")
        try:
            np.save(tmp_path, tensor.detach().cpu().numpy().astype(np.float32, copy=False))
            os.replace(tmp_path, path)
        except OSError:
            if path.exists():
                try:
                    tmp_path.unlink(missing_ok=True)
                except OSError:
                    pass
                return
            raise

    def _preprocess_uncached(self, record: AlanRecord) -> torch.Tensor:
        volume = np.load(record.volume_path).astype(np.float32, copy=False)
        volume = apply_nan_strategy(volume, self.nan_strategy, self.nan_fill_value)
        if self.use_bbox_crop:
            volume = crop_to_bbox(
                volume=volume,
                bbox_min=record.bbox_min,
                bbox_max=record.bbox_max,
                margin=self.bbox_margin,
            )
        if self.canonicalize_right and record.side == "R":
            volume = np.flip(volume, axis=self.right_flip_axis).copy()
        if self.pad_to_cube_input:
            volume = pad_to_cube(volume)
        tensor = resize_volume(volume, self.target_shape)
        return tensor.clamp_(0.0, 1.0)

    def _preprocess_base(self, record: AlanRecord) -> torch.Tensor:
        if self.cache_mode == "none":
            return self._preprocess_uncached(record)

        key = self._cache_key(record)
        if self.cache_mode == "memory":
            cached = self._memory_cache.get(key)
            if cached is None:
                cached = self._preprocess_uncached(record)
                self._memory_cache[key] = cached
            return cached

        cached = self._load_disk_cache(key)
        if cached is not None:
            return cached

        tensor = self._preprocess_uncached(record)
        self._save_disk_cache(key, tensor)
        return tensor

    def _preprocess(self, record: AlanRecord) -> torch.Tensor:
        tensor = self._preprocess_base(record)
        if self.cache_mode == "memory" or self.transform is not None:
            tensor = tensor.clone()
        if self.transform is not None:
            tensor = self.transform(tensor)
            tensor = tensor.clamp_(0.0, 1.0)
        return tensor

    def __getitem__(self, index: int) -> dict[str, object]:
        record = self.records[index]
        volume = self._preprocess(record)
        return {
            "id": record.roi_id,
            "volume": volume,
            "label": torch.tensor(record.label, dtype=torch.float32),
            "side": record.side,
            "subset": record.subset,
            "voxel_count": torch.tensor(record.voxel_count, dtype=torch.float32),
        }


@lru_cache(maxsize=1)
def _preprocess_cache_fingerprint() -> str:
    objects = (
        apply_nan_strategy,
        crop_to_bbox,
        pad_to_cube,
        resize_volume,
        AlanKidneyDataset._preprocess_uncached,
    )
    sources = []
    for obj in objects:
        try:
            sources.append(inspect.getsource(obj))
        except (OSError, TypeError):
            sources.append(repr(obj))
    raw = "\n".join(sources).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()
