"""Backend-facing inference bridge that wraps the ClassificationV2 stack.

The NestJS backend spawns ``python/predict.py`` which imports
``DummyFrontendBridge`` from this module. The bridge:

1. Loads a preview manifest mapping a known .npy volume to a small fingerprint.
2. Fingerprint-matches the uploaded preview image to find the corresponding ROI.
3. Loads the matched .npy volume and applies V2 training preprocessing.
4. Builds a V2 model, loads a checkpoint, and runs binary-sigmoid inference.
5. Optionally applies temperature/isotonic calibration from the run directory.
6. Returns a dict with the contract expected by ``python/predict.py``.

All diagnostic output goes to stderr — stdout belongs to the caller.
"""
from __future__ import annotations

import csv
import json
import os
import re
import sys
import traceback
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

_HERE = Path(__file__).resolve()
_CLASSIFICATION_ROOT = _HERE.parents[2]

# Ensure top-level ClassificationV2 packages (Model/, Preprocessing/, Utils/)
# are importable regardless of who launched us.
if str(_CLASSIFICATION_ROOT) not in sys.path:
    sys.path.insert(0, str(_CLASSIFICATION_ROOT))

from Model.factory import build_model  # noqa: E402
from Preprocessing.dataset import (  # noqa: E402
    apply_nan_strategy,
    crop_to_bbox,
    pad_to_cube,
    resize_volume,
)
from Utils.config import DataConfig, ModelConfig  # noqa: E402

try:  # Calibration is optional; tolerate missing helpers.
    from Utils.calibration import IsotonicResult, apply_isotonic, apply_temperature
except ImportError:  # pragma: no cover - defensive
    IsotonicResult = None  # type: ignore[assignment]
    apply_isotonic = None  # type: ignore[assignment]
    apply_temperature = None  # type: ignore[assignment]

from src.utils.fingerprint import (  # noqa: E402
    fingerprint_score,
    load_fingerprint,
    match_confidence as _match_confidence,
)


TABULAR_FEATURE_NAMES = ("log_voxel_count_z", "side_is_left")


def _log(message: str) -> None:
    print(f"[bridge] {message}", file=sys.stderr, flush=True)


def _resolve_path(value: str | Path | None, root: Path) -> Path | None:
    """Resolve a possibly-relative path against the Classification root.

    Handles three input flavours so users don't have to remember which one is
    correct:

    * absolute paths (Windows or POSIX) are returned as-is,
    * paths starting with ``Classification/`` (or its backslash variant) are
      treated as relative to the *parent* of the Classification root so the
      leading segment is not duplicated,
    * anything else is treated as relative to the Classification root.
    """
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    raw = raw.replace("\\", "/").lstrip("./")
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    parts = candidate.parts
    if parts and parts[0].lower() == root.name.lower():
        return (root.parent / candidate).resolve()
    return (root / candidate).resolve()


def _coerce_dataclass(cls, payload: dict[str, Any]):
    if not is_dataclass(cls):
        return cls(**payload) if payload else cls()
    valid_names = {f.name for f in fields(cls)}
    filtered = {key: value for key, value in payload.items() if key in valid_names}
    instance = cls(**filtered)
    return instance


_BOOTSTRAP_RUN_NAME = "predictor_v2"


def _select_v2_run_dir(root: Path) -> Path | None:
    """Pick a V2 run directory.

    Priority:
      1. ``PREDICTOR_V2_RUN_DIR`` (resolved with the same rules as other paths).
      2. Real runs containing both ``best_model.pt`` and ``config.json``.
      3. Anything containing ``best_model.pt``.

    The legacy bootstrap directory ``outputs/predictor_v2`` is deprioritised:
    it only wins when no real run is present.
    """
    explicit = os.environ.get("PREDICTOR_V2_RUN_DIR")
    if explicit:
        candidate = _resolve_path(explicit, root)
        if candidate is not None and candidate.is_dir():
            _log(f"selected V2 run dir (env): {candidate}")
            return candidate
        _log(f"PREDICTOR_V2_RUN_DIR is set but not a usable directory: {explicit}")

    outputs = root / "outputs"
    if not outputs.is_dir():
        return None

    candidates: set[Path] = set()
    for path in outputs.rglob("best_model.pt"):
        candidates.add(path.parent)

    if not candidates:
        return None

    def _score(directory: Path) -> tuple[int, int, float]:
        has_config = (directory / "config.json").exists()
        is_bootstrap = directory.name == _BOOTSTRAP_RUN_NAME
        # Higher tuple wins: prefer (real run) over (bootstrap), prefer dirs
        # with config.json, then most-recently-modified.
        return (
            0 if is_bootstrap else 1,
            1 if has_config else 0,
            directory.stat().st_mtime,
        )

    ranked = sorted(candidates, key=_score, reverse=True)
    selected = ranked[0]
    if selected.name == _BOOTSTRAP_RUN_NAME:
        _log(
            "WARNING: only the bootstrap random run is available; "
            "predictions will be untrustworthy."
        )
    _log(f"selected V2 run dir (auto): {selected}")
    return selected


def _load_metadata(metadata_csv: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not metadata_csv.exists():
        return rows
    with metadata_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            roi_id = row.get("ROI_id")
            if roi_id:
                rows[roi_id] = row
    return rows


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _to_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strip_module_prefix(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor] | None:
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            inner = payload.get(key)
            if isinstance(inner, dict):
                return _strip_module_prefix(inner)
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return _strip_module_prefix(payload)
    return None


class DummyFrontendBridge:
    def __init__(
        self,
        manifest_path: str | Path,
        checkpoint_path: str | Path,
        config: dict[str, Any] | None = None,
        device: str | None = None,
        threshold: float | None = None,
    ) -> None:
        self.root = _CLASSIFICATION_ROOT
        self._explicit_threshold = threshold
        self._yaml_config = config if isinstance(config, dict) else None

        self.manifest_path = self._resolve_manifest_path(manifest_path)
        self.manifest = self._load_manifest(self.manifest_path)
        self.fingerprint_size = self._extract_fingerprint_size(self.manifest)

        self.run_dir = _select_v2_run_dir(self.root)

        self.run_config = self._load_run_config(self.run_dir)
        self.calibration = self._load_calibration(self.run_dir)

        self.data_config = self._build_data_config(self.run_config)
        self.model_config = self._build_model_config(self.run_config)

        self.device = self._resolve_device(device)
        _log(f"device: {self.device}")

        self.checkpoint_path = self._resolve_checkpoint_path(checkpoint_path)
        _log(f"checkpoint: {self.checkpoint_path}")

        self.tabular_stats = self._extract_tabular_stats(self.run_config)
        self.metadata = _load_metadata(self.root / Path(str(self.data_config.metadata_csv)))

        self.threshold = self._resolve_threshold()
        _log(f"decision threshold: {self.threshold:.4f}")

        self.model = self._build_model_and_load_weights()

    # ------------------------------------------------------------------
    # path & config helpers
    # ------------------------------------------------------------------
    def _resolve_manifest_path(self, manifest_path: str | Path) -> Path:
        env_override = os.environ.get("PREDICTOR_MANIFEST_PATH")
        if env_override:
            manifest_path = env_override
        resolved = _resolve_path(manifest_path, self.root)
        if resolved is None or not resolved.exists():
            fallback = self.root / "outputs" / "jpg_exports_fixed" / "preview_manifest.json"
            if fallback.exists():
                _log(f"manifest fallback: {fallback}")
                return fallback
            raise FileNotFoundError(
                f"Preview manifest not found. Tried: {resolved} and {fallback}"
            )
        return resolved

    def _load_manifest(self, manifest_path: Path) -> dict[str, Any]:
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        entries = payload.get("entries")
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Preview manifest has no entries: {manifest_path}")
        return payload

    @staticmethod
    def _extract_fingerprint_size(manifest: dict[str, Any]) -> tuple[int, int]:
        size = manifest.get("fingerprint_size") or [16, 16]
        return (int(size[0]), int(size[1]))

    def _load_run_config(self, run_dir: Path | None) -> dict[str, Any]:
        if run_dir is None:
            return {}
        config_path = run_dir / "config.json"
        if not config_path.exists():
            _log(f"run dir has no config.json: {config_path}")
            return {}
        try:
            with config_path.open("r", encoding="utf-8") as handle:
                return json.load(handle)
        except Exception as exc:  # noqa: BLE001
            _log(f"failed to read {config_path}: {exc}")
            return {}

    def _load_calibration(self, run_dir: Path | None) -> dict[str, Any]:
        if run_dir is None:
            return {}
        path = run_dir / "calibration.json"
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:  # noqa: BLE001
            _log(f"failed to read {path}: {exc}")
            return {}
        return payload if isinstance(payload, dict) else {}

    def _build_data_config(self, run_config: dict[str, Any]) -> DataConfig:
        raw = run_config.get("data") if isinstance(run_config, dict) else None
        if isinstance(raw, dict):
            data_config = _coerce_dataclass(DataConfig, raw)
        else:
            data_config = DataConfig()
        return data_config.resolved()

    def _build_model_config(self, run_config: dict[str, Any]) -> ModelConfig:
        raw = run_config.get("model") if isinstance(run_config, dict) else None
        if isinstance(raw, dict):
            return _coerce_dataclass(ModelConfig, raw)
        return ModelConfig()

    def _extract_tabular_stats(self, run_config: dict[str, Any]) -> dict[str, float] | None:
        if not getattr(self.model_config, "use_tabular_features", False):
            return None
        stats = run_config.get("tabular_feature_stats") if isinstance(run_config, dict) else None
        if isinstance(stats, dict):
            try:
                return {
                    "log_voxel_count_mean": float(stats["log_voxel_count_mean"]),
                    "log_voxel_count_std": float(stats["log_voxel_count_std"]) or 1.0,
                }
            except (KeyError, TypeError, ValueError):
                _log("tabular_feature_stats malformed; falling back to identity stats")
        # Last-resort: identity stats (z = log1p(voxel_count) directly)
        return {"log_voxel_count_mean": 0.0, "log_voxel_count_std": 1.0}

    def _resolve_device(self, device: str | None) -> torch.device:
        env_device = os.environ.get("PREDICTOR_DEVICE")
        request = device or env_device or "auto"
        if request == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            return torch.device("cpu")
        return torch.device(request)

    def _resolve_checkpoint_path(self, checkpoint_path: str | Path) -> Path:
        candidates: list[Path] = []
        explicit = _resolve_path(checkpoint_path, self.root)
        if explicit is not None:
            candidates.append(explicit)
        if self.run_dir is not None:
            candidates.append(self.run_dir / "best_model.pt")
            candidates.append(self.run_dir / "best_run" / "best_model.pt")

        for path in candidates:
            if path.exists():
                return path

        searched = "\n  ".join(str(c) for c in candidates) or "(none)"
        raise FileNotFoundError(
            "No usable model checkpoint found. Tried:\n  " + searched
        )

    def _resolve_threshold(self) -> float:
        if self._explicit_threshold is not None:
            return float(self._explicit_threshold)
        env = os.environ.get("PREDICTOR_THRESHOLD")
        if env:
            try:
                return float(env)
            except ValueError:
                _log(f"PREDICTOR_THRESHOLD={env!r} is not a float; ignoring")
        for source in (self.run_config, self.calibration):
            if not isinstance(source, dict):
                continue
            for key in ("optimal_threshold", "tuned_threshold", "fixed_threshold"):
                value = source.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
        return 0.5

    # ------------------------------------------------------------------
    # model / inference
    # ------------------------------------------------------------------
    def _build_model_and_load_weights(self) -> torch.nn.Module:
        num_tabular = (
            len(TABULAR_FEATURE_NAMES)
            if getattr(self.model_config, "use_tabular_features", False)
            else 0
        )
        model = build_model(model_config=self.model_config, num_tabular_features=num_tabular)
        try:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        except Exception:
            payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = _extract_state_dict(payload)
        if state_dict is None:
            raise RuntimeError(
                f"Checkpoint at {self.checkpoint_path} does not contain a recognisable state_dict."
            )
        # Filter out keys that look like serialized auxiliary metadata rather
        # than model weights (e.g. saved tracking buffers from optimizers).
        ignorable_suffixes = ("num_batches_tracked",)
        unexpected_safe: list[str] = []
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=True)
        except RuntimeError:
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            unexpected_safe = [k for k in unexpected if not k.endswith(ignorable_suffixes)]
            if missing or unexpected_safe:
                raise RuntimeError(
                    "Checkpoint does not match model architecture.\n"
                    f"  checkpoint: {self.checkpoint_path}\n"
                    f"  missing keys ({len(missing)}): {missing[:5]}\n"
                    f"  unexpected keys ({len(unexpected_safe)}): {unexpected_safe[:5]}"
                )
            _log(
                "ignored harmless checkpoint keys: "
                f"{[k for k in unexpected if k.endswith(ignorable_suffixes)][:5]}"
            )
        model.eval()
        model.to(self.device)
        return model

    # ------------------------------------------------------------------
    # public entry point
    # ------------------------------------------------------------------
    def predict_from_image(self, image_path: str) -> dict[str, Any]:
        match = self._match_image(image_path)
        roi_id = match["roi_id"]
        npy_path = match["npy_path"]
        score = match["score"]
        confidence = match["confidence"]

        volume_tensor = self._preprocess_volume(npy_path, roi_id)
        tabular = self._build_tabular_features(roi_id)

        with torch.inference_mode():
            inputs = volume_tensor.to(self.device, non_blocking=False)
            if getattr(self.model_config, "use_tabular_features", False):
                logits = self.model(inputs, tabular_features=tabular.to(self.device))
            else:
                logits = self.model(inputs, tabular_features=None)

        logit_value = float(logits.detach().cpu().reshape(-1)[0].item())
        probability_anomaly = self._calibrated_probability(logit_value)
        probability_normal = float(np.clip(1.0 - probability_anomaly, 0.0, 1.0))
        predicted_label = "anomaly" if probability_anomaly >= self.threshold else "normal"

        return {
            "matched_roi_id": roi_id,
            "matched_npy_path": str(npy_path),
            "match_score": float(score),
            "match_confidence": float(confidence),
            "prediction": {
                "predicted_label": predicted_label,
                "probability_anomaly": float(probability_anomaly),
                "probability_normal": float(probability_normal),
            },
        }

    # ------------------------------------------------------------------
    # matching / preprocessing
    # ------------------------------------------------------------------
    def _match_image(self, image_path: str) -> dict[str, Any]:
        query = load_fingerprint(image_path, size=self.fingerprint_size)
        best: dict[str, Any] | None = None
        for entry in self.manifest["entries"]:
            ref = entry.get("fingerprint")
            if not isinstance(ref, list) or not ref:
                continue
            ref_arr = np.asarray(ref, dtype=np.float32)
            if ref_arr.shape != query.shape:
                continue
            score = fingerprint_score(query, ref_arr)
            if best is None or score < best["score"]:
                best = {
                    "roi_id": entry.get("roi_id"),
                    "npy_path": _resolve_path(entry.get("npy_path"), self.root),
                    "score": score,
                    "confidence": _match_confidence(score),
                }
        if best is None or best.get("npy_path") is None:
            raise RuntimeError("No manifest entry matched the uploaded preview.")
        npy_path: Path = best["npy_path"]  # type: ignore[assignment]
        if not npy_path.exists():
            raise FileNotFoundError(f"Matched .npy volume not found: {npy_path}")
        return best

    def _preprocess_volume(self, npy_path: Path, roi_id: str) -> torch.Tensor:
        volume = np.load(npy_path).astype(np.float32, copy=False)
        volume = apply_nan_strategy(
            volume,
            self.data_config.nan_strategy if self.data_config.nan_strategy != "drop_record" else "none",
            float(self.data_config.nan_fill_value),
        )

        meta = self.metadata.get(roi_id, {})
        side = (meta.get("side") or "").upper() or "L"

        if self.data_config.use_bbox_crop and meta:
            bbox_min = (
                _to_int(meta.get("bbox_min_d"), 0),
                _to_int(meta.get("bbox_min_h"), 0),
                _to_int(meta.get("bbox_min_w"), 0),
            )
            bbox_max = (
                _to_int(meta.get("bbox_max_d"), volume.shape[0] - 1),
                _to_int(meta.get("bbox_max_h"), volume.shape[1] - 1),
                _to_int(meta.get("bbox_max_w"), volume.shape[2] - 1),
            )
            volume = crop_to_bbox(
                volume=volume,
                bbox_min=bbox_min,
                bbox_max=bbox_max,
                margin=int(self.data_config.bbox_margin),
            )

        if self.data_config.canonicalize_right and side == "R":
            volume = np.flip(volume, axis=int(self.data_config.right_flip_axis)).copy()

        if self.data_config.pad_to_cube:
            volume = pad_to_cube(volume)

        tensor = resize_volume(volume, tuple(self.data_config.target_shape))
        tensor = tensor.clamp_(0.0, 1.0)
        # resize_volume returns shape [C, D, H, W]; add a batch dim -> [1, C, D, H, W]
        return tensor.unsqueeze(0)

    def _build_tabular_features(self, roi_id: str) -> torch.Tensor:
        if not getattr(self.model_config, "use_tabular_features", False):
            return torch.empty(0)
        meta = self.metadata.get(roi_id, {})
        voxel_count = _to_float(meta.get("voxel_count"), 0.0)
        side = (meta.get("side") or "L").upper()
        stats = self.tabular_stats or {"log_voxel_count_mean": 0.0, "log_voxel_count_std": 1.0}
        std = float(stats.get("log_voxel_count_std", 1.0)) or 1.0
        mean = float(stats.get("log_voxel_count_mean", 0.0))
        log_voxel_z = (np.log1p(max(voxel_count, 0.0)) - mean) / std
        side_is_left = 1.0 if side == "L" else 0.0
        return torch.tensor([[float(log_voxel_z), side_is_left]], dtype=torch.float32)

    # ------------------------------------------------------------------
    # calibration
    # ------------------------------------------------------------------
    def _calibrated_probability(self, logit_value: float) -> float:
        temperature = None
        for source in (self.run_config, self.calibration):
            if isinstance(source, dict) and isinstance(source.get("temperature"), (int, float)):
                temperature = float(source["temperature"])
                break
        if not temperature or temperature <= 0.0 or not np.isfinite(temperature):
            temperature = 1.0

        scaled = logit_value / temperature
        prob = 1.0 / (1.0 + float(np.exp(-scaled)))

        iso_x = self.calibration.get("isotonic_x") if isinstance(self.calibration, dict) else None
        iso_y = self.calibration.get("isotonic_y") if isinstance(self.calibration, dict) else None
        if isinstance(iso_x, list) and isinstance(iso_y, list) and iso_x and iso_y and len(iso_x) == len(iso_y):
            try:
                xs = np.asarray(iso_x, dtype=np.float64)
                ys = np.asarray(iso_y, dtype=np.float64)
                clipped = float(np.clip(prob, xs[0], xs[-1]))
                prob = float(np.clip(np.interp(clipped, xs, ys), 0.0, 1.0))
            except Exception as exc:  # noqa: BLE001
                _log(f"isotonic application failed; using temperature-only prob ({exc})")

        return float(np.clip(prob, 0.0, 1.0))


__all__ = ["DummyFrontendBridge"]
