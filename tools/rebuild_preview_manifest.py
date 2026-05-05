"""Rebuild preview manifest paths for the renamed Classification root.

The legacy manifest baked absolute D:\\... paths from a previous machine.
This script rewrites every entry so paths are relative to the Classification
root and resolvable in the new layout (npy under ``ALAN/alan``, previews
under ``outputs/jpg_exports_fixed``). Entries whose .npy or preview file
cannot be found are dropped, with a stderr summary.

Run from inside Classification/.venv:
    python tools/rebuild_preview_manifest.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def _classification_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _resolve_existing(root: Path, candidates: list[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.is_absolute() and candidate.exists():
            return candidate
        relative = (root / candidate).resolve()
        if relative.exists():
            return relative
    return None


def _normalize_relative(root: Path, target: Path) -> str:
    try:
        return target.resolve().relative_to(root).as_posix()
    except ValueError:
        return target.resolve().as_posix()


def main() -> int:
    root = _classification_root()
    manifest_path = root / "outputs" / "jpg_exports_fixed" / "preview_manifest.json"
    if not manifest_path.exists():
        print(f"manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = payload.get("entries", [])
    if not isinstance(entries, list):
        print("manifest 'entries' is not a list", file=sys.stderr)
        return 1

    rewritten: list[dict] = []
    dropped = 0
    for entry in entries:
        roi_id = entry.get("roi_id")
        if not roi_id:
            dropped += 1
            continue

        npy_candidates: list[Path] = []
        npy_rel = entry.get("npy_rel_path")
        if isinstance(npy_rel, str):
            npy_candidates.append(Path(npy_rel.replace("\\", "/")))
        npy_abs = entry.get("npy_path")
        if isinstance(npy_abs, str):
            npy_candidates.append(Path(npy_abs.replace("\\", "/")))
        npy_candidates.append(Path("ALAN") / "alan" / f"{roi_id}.npy")

        npy_resolved = _resolve_existing(root, npy_candidates)
        if npy_resolved is None:
            dropped += 1
            continue

        preview_candidates: list[Path] = []
        preview_rel = entry.get("preview_rel_path")
        if isinstance(preview_rel, str):
            preview_candidates.append(Path(preview_rel.replace("\\", "/")))
        preview_abs = entry.get("preview_path")
        if isinstance(preview_abs, str):
            preview_candidates.append(Path(preview_abs.replace("\\", "/")))

        preview_resolved = _resolve_existing(root, preview_candidates)
        if preview_resolved is None:
            dropped += 1
            continue

        new_entry = dict(entry)
        new_entry["npy_path"] = _normalize_relative(root, npy_resolved)
        new_entry["preview_path"] = _normalize_relative(root, preview_resolved)
        new_entry.pop("npy_rel_path", None)
        new_entry.pop("preview_rel_path", None)
        rewritten.append(new_entry)

    payload["entries"] = rewritten
    payload["dataset_path"] = "ALAN"
    payload["image_source_path"] = "ALAN/alan"
    payload["output_dir"] = "outputs/jpg_exports_fixed"
    if "fingerprint_size" not in payload:
        payload["fingerprint_size"] = [16, 16]

    manifest_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(
        f"rewrote {len(rewritten)} entries, dropped {dropped} "
        f"-> {manifest_path}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
