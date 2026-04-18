from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def _compute_bbox(mask: np.ndarray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    foreground = np.argwhere(mask)
    if foreground.size == 0:
        max_index = tuple(int(size - 1) for size in mask.shape)
        return (0, 0, 0), max_index
    bbox_min = tuple(int(value) for value in foreground.min(axis=0))
    bbox_max = tuple(int(value) for value in foreground.max(axis=0))
    return bbox_min, bbox_max


def build_metadata(info_csv: Path, volumes_dir: Path) -> tuple[list[dict[str, str | int | float]], dict[str, object]]:
    records: list[dict[str, str | int | float]] = []
    subset_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter()
    bbox_sizes: list[tuple[int, int, int]] = []
    volume_sums: list[int] = []
    side_counts: Counter[str] = Counter()

    with info_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            roi_id = row["ROI_id"]
            subset = row["subset"]
            label_text = row["ROI_anomaly"]
            label_int = int(label_text.upper() == "TRUE")
            side = roi_id.split("_")[-1]
            volume_path = volumes_dir / f"{roi_id}.npy"
            if not volume_path.exists():
                raise FileNotFoundError(f"Missing volume for ROI_id={roi_id}: {volume_path}")

            volume = np.load(volume_path)
            bbox_min, bbox_max = _compute_bbox(volume)
            bbox_size = tuple((bbox_max[index] - bbox_min[index] + 1) for index in range(3))
            center = np.argwhere(volume).mean(axis=0) if volume.any() else np.array([0.0, 0.0, 0.0])
            voxel_count = int(np.count_nonzero(volume))
            nan_count = int(np.isnan(volume.astype(np.float32, copy=False)).sum())
            nan_ratio = nan_count / max(volume.size, 1)
            has_nan = nan_count > 0

            subset_counts.update([subset])
            label_counts.update([label_text])
            side_counts.update([side])
            bbox_sizes.append(bbox_size)
            volume_sums.append(voxel_count)

            records.append(
                {
                    "ROI_id": roi_id,
                    "subset": subset,
                    "ROI_anomaly": label_text,
                    "label_int": label_int,
                    "side": side,
                    "volume_path": volume_path.name,
                    "voxel_count": voxel_count,
                    "bbox_min_d": bbox_min[0],
                    "bbox_min_h": bbox_min[1],
                    "bbox_min_w": bbox_min[2],
                    "bbox_max_d": bbox_max[0],
                    "bbox_max_h": bbox_max[1],
                    "bbox_max_w": bbox_max[2],
                    "center_d": float(center[0]),
                    "center_h": float(center[1]),
                    "center_w": float(center[2]),
                    "shape_d": int(volume.shape[0]),
                    "shape_h": int(volume.shape[1]),
                    "shape_w": int(volume.shape[2]),
                    "nan_count": nan_count,
                    "nan_ratio": round(nan_ratio, 8),
                    "has_nan": int(has_nan),
                }
            )

    # Patient-level split consistency check
    patient_subsets: dict[str, set[str]] = {}
    for rec in records:
        patient_id = rec["ROI_id"].rsplit("_", 1)[0]
        patient_subsets.setdefault(patient_id, set()).add(rec["subset"])
    leaking = {pid: subs for pid, subs in patient_subsets.items() if len(subs) > 1}
    if leaking:
        raise ValueError(f"Patient-level data leakage detected! Patients in multiple splits: {leaking}")

    # Per-split class distribution
    split_label_counts: dict[str, dict[str, int]] = {}
    for rec in records:
        sub = rec["subset"]
        lab = rec["ROI_anomaly"]
        split_label_counts.setdefault(sub, Counter()).update([lab])
    split_label_counts = {k: dict(v) for k, v in split_label_counts.items()}

    # NaN summary
    nan_samples = [rec for rec in records if rec["has_nan"]]
    total_nan_voxels = sum(rec["nan_count"] for rec in records)
    split_nan_counts: dict[str, int] = {}
    for rec in nan_samples:
        split_nan_counts[rec["subset"]] = split_nan_counts.get(rec["subset"], 0) + 1

    bbox_array = np.asarray(bbox_sizes, dtype=np.float64)
    volume_array = np.asarray(volume_sums, dtype=np.float64)
    summary = {
        "samples": len(records),
        "patients": len(patient_subsets),
        "subset_counts": dict(subset_counts),
        "label_counts": dict(label_counts),
        "split_label_counts": split_label_counts,
        "side_counts": dict(side_counts),
        "bbox_mean": bbox_array.mean(axis=0).round(4).tolist(),
        "bbox_p95": np.percentile(bbox_array, 95, axis=0).round(4).tolist(),
        "voxel_count_mean": float(volume_array.mean()),
        "voxel_count_std": float(volume_array.std()),
        "voxel_count_min": int(volume_array.min()),
        "voxel_count_max": int(volume_array.max()),
        "nan_samples": len(nan_samples),
        "nan_total_voxels": total_nan_voxels,
        "nan_split_counts": split_nan_counts,
    }
    return records, summary


def write_metadata(records: list[dict[str, str | int | float]], metadata_csv: Path) -> None:
    metadata_csv.parent.mkdir(parents=True, exist_ok=True)
    with metadata_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)


def write_summary(summary: dict[str, object], summary_json: Path) -> None:
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2))


_REQUIRED_COLUMNS = {"nan_count", "nan_ratio", "has_nan"}


def ensure_metadata(info_csv: Path, volumes_dir: Path, metadata_csv: Path, summary_json: Path | None = None) -> Path:
    if metadata_csv.exists():
        with metadata_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            existing_columns = set(reader.fieldnames or [])
        if _REQUIRED_COLUMNS.issubset(existing_columns):
            return metadata_csv
    records, summary = build_metadata(info_csv=info_csv, volumes_dir=volumes_dir)
    write_metadata(records, metadata_csv)
    if summary_json is not None:
        write_summary(summary, summary_json)
    return metadata_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the ALAN dataset and create metadata.")
    parser.add_argument("--info-csv", type=Path, default=Path("ALAN/info.csv"))
    parser.add_argument("--volumes-dir", type=Path, default=Path("ALAN/alan"))
    parser.add_argument("--metadata-csv", type=Path, default=Path("ALAN/metadata.csv"))
    parser.add_argument("--summary-json", type=Path, default=Path("ALAN/summary.json"))
    args = parser.parse_args()

    records, summary = build_metadata(info_csv=args.info_csv, volumes_dir=args.volumes_dir)
    write_metadata(records, args.metadata_csv)
    write_summary(summary, args.summary_json)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

