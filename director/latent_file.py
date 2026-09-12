"""Portable single-latent storage for the standalone H3 workflow nodes.

The official ComfyUI ``SaveLatent`` node stores one regular tensor in a
``.latent`` safetensors file.  MiniMax H3 uses an AV ``NestedTensor`` (video
and audio are separate tensors), so the standalone nodes use a small zip
container with one safetensors file per stream.  No pickle is used.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import stat
import tempfile
import uuid
from pathlib import Path
from typing import Any
from zipfile import ZIP_STORED, ZipFile

import torch

from .h3_motion_context import _repack_av_streams, _streams_from_latent

FORMAT_NAME = "minimax-h3-av-latent"
FORMAT_VERSION = 1
FILE_SUFFIX = ".mmxlatent.zip"
_MAX_ARCHIVE_ENTRIES = 128
_MAX_ARCHIVE_MEMBER_BYTES = 16 * 1024 * 1024 * 1024
_MAX_ARCHIVE_TOTAL_BYTES = 32 * 1024 * 1024 * 1024
_UNSERIALIZABLE = object()


def _safetensor_io():
    try:
        from safetensors.torch import load_file, save_file
    except Exception as exc:  # pragma: no cover - depends on ComfyUI install
        raise RuntimeError(
            "MiniMax H3 latent storage requires the 'safetensors' package."
        ) from exc
    return load_file, save_file


def _json_safe(value: Any):
    """Return a JSON-compatible copy, or a private sentinel when unsupported."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            safe = _json_safe(item)
            if safe is not _UNSERIALIZABLE:
                out[str(key)] = safe
        return out
    if isinstance(value, (list, tuple)):
        out = []
        for item in value:
            safe = _json_safe(item)
            if safe is _UNSERIALIZABLE:
                return _UNSERIALIZABLE
            out.append(safe)
        return out
    return _UNSERIALIZABLE


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    _load_file, save_file = _safetensor_io()
    del _load_file
    save_file(
        {"tensor": tensor.detach().cpu().contiguous()},
        str(path),
        metadata={"format": FORMAT_NAME, "version": str(FORMAT_VERSION)},
    )


def _load_tensor(path: Path) -> torch.Tensor:
    load_file, _save_file = _safetensor_io()
    tensors = load_file(str(path), device="cpu")
    if set(tensors) != {"tensor"}:
        raise ValueError(f"Invalid H3 latent tensor file: {path.name}")
    return tensors["tensor"]


def _validate_streams(latent: dict) -> list[torch.Tensor]:
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("Expected a LATENT dictionary with a 'samples' field.")
    streams = _streams_from_latent(latent)
    if len(streams) < 1:
        raise ValueError("MiniMax H3 latent has no streams to save.")
    if not all(torch.is_tensor(stream) for stream in streams):
        raise ValueError("MiniMax H3 latent streams must be torch tensors.")
    return [stream.detach().cpu().contiguous() for stream in streams]


def _archive_member_name(name: str) -> str:
    """Reject traversal, absolute paths, and symlink-like archive members."""
    normalized = str(name).replace("\\", "/")
    if not normalized or normalized.startswith("/") or re.match(r"^[A-Za-z]:", normalized):
        raise ValueError(f"Invalid latent archive member: {name!r}")
    parts = normalized.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(f"Invalid latent archive member: {name!r}")
    if any(ord(char) < 32 for char in normalized):
        raise ValueError(f"Invalid latent archive member: {name!r}")
    return "/".join(parts)


def _extract_archive(archive: Path, destination: Path) -> set[str]:
    names: set[str] = set()
    total_size = 0
    with ZipFile(archive, "r") as zf:
        infos = zf.infolist()
        if len(infos) > _MAX_ARCHIVE_ENTRIES:
            raise ValueError("Latent archive contains too many files.")
        for info in infos:
            if info.is_dir():
                continue
            name = _archive_member_name(info.filename)
            if name in names:
                raise ValueError(f"Duplicate latent archive member: {name}")
            names.add(name)
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError("Symlink members are not allowed in latent archives.")
            if info.file_size > _MAX_ARCHIVE_MEMBER_BYTES:
                raise ValueError(f"Latent archive member is too large: {name}")
            total_size += int(info.file_size)
            if total_size > _MAX_ARCHIVE_TOTAL_BYTES:
                raise ValueError("Latent archive expands beyond the allowed size.")
            target = destination.joinpath(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info, "r") as source, target.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
    return names


def _read_json(path: Path, description: str):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"Invalid {description} in H3 latent archive.") from exc


def _validate_manifest(manifest: Any, names: set[str]) -> dict:
    if not isinstance(manifest, dict):
        raise ValueError("H3 latent manifest must be an object.")
    if manifest.get("format") != FORMAT_NAME:
        raise ValueError("This file is not a MiniMax H3 AV latent archive.")
    if int(manifest.get("version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported MiniMax H3 latent archive version.")

    streams = manifest.get("streams")
    if not isinstance(streams, list) or not streams:
        raise ValueError("H3 latent archive contains no stream manifest.")
    for item in streams:
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid H3 latent stream manifest.")
        path = _archive_member_name(item["path"])
        if path not in names or not path.startswith("streams/"):
            raise ValueError("H3 latent stream file is missing.")

    for item in manifest.get("tensor_extras", []):
        if not isinstance(item, dict) or not isinstance(item.get("path"), str):
            raise ValueError("Invalid H3 latent tensor-extra manifest.")
        path = _archive_member_name(item["path"])
        if path not in names or not path.startswith("tensor_extras/"):
            raise ValueError("H3 latent tensor-extra file is missing.")

    json_path = manifest.get("json_extras_path")
    if json_path is not None:
        json_path = _archive_member_name(json_path)
        if json_path not in names:
            raise ValueError("H3 latent JSON extras file is missing.")

    metadata_path = manifest.get("metadata_path")
    if metadata_path is not None:
        metadata_path = _archive_member_name(metadata_path)
        if metadata_path not in names:
            raise ValueError("H3 latent metadata file is missing.")
    return manifest


def save_h3_latent(
    latent: dict,
    destination: str | Path,
    *,
    metadata: dict | None = None,
) -> dict:
    """Atomically save an H3 AV latent and return a small file summary."""
    streams = _validate_streams(latent)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=".mmxlatent-", dir=str(destination.parent)))
    archive_tmp = destination.with_name(
        f".{destination.name}.{uuid.uuid4().hex}.tmp"
    )
    skipped: list[str] = []
    try:
        stream_entries = []
        for index, stream in enumerate(streams):
            relative = f"streams/{index:02d}.safetensors"
            (staging / relative).parent.mkdir(parents=True, exist_ok=True)
            _save_tensor(staging / relative, stream)
            stream_entries.append(
                {
                    "path": relative,
                    "shape": list(stream.shape),
                    "dtype": str(stream.dtype),
                }
            )

        tensor_entries = []
        json_extras = {}
        for key, value in latent.items():
            key = str(key)
            if key in {"samples", "mmx_latent_metadata"}:
                continue
            if torch.is_tensor(value):
                relative = f"tensor_extras/{len(tensor_entries):02d}.safetensors"
                (staging / relative).parent.mkdir(parents=True, exist_ok=True)
                _save_tensor(staging / relative, value)
                tensor_entries.append({"key": key, "path": relative})
                continue
            safe = _json_safe(value)
            if safe is _UNSERIALIZABLE:
                skipped.append(key)
            else:
                json_extras[key] = safe

        manifest: dict[str, Any] = {
            "format": FORMAT_NAME,
            "version": FORMAT_VERSION,
            "streams": stream_entries,
            "tensor_extras": tensor_entries,
            "skipped_extras": skipped,
        }
        if json_extras:
            json_path = "json_extras.json"
            (staging / json_path).write_text(
                json.dumps(json_extras, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            manifest["json_extras_path"] = json_path

        merged_metadata = dict(metadata or {})
        embedded_metadata = latent.get("mmx_latent_metadata")
        if isinstance(embedded_metadata, dict):
            for key, value in embedded_metadata.items():
                merged_metadata.setdefault(str(key), value)
        safe_metadata = _json_safe(merged_metadata)
        if safe_metadata is not _UNSERIALIZABLE and safe_metadata:
            metadata_path = "metadata.json"
            (staging / metadata_path).write_text(
                json.dumps(safe_metadata, ensure_ascii=False, separators=(",", ":")),
                encoding="utf-8",
            )
            manifest["metadata_path"] = metadata_path

        (staging / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        with ZipFile(archive_tmp, "w", compression=ZIP_STORED) as zf:
            for file_path in sorted(staging.rglob("*")):
                if file_path.is_file():
                    zf.write(file_path, file_path.relative_to(staging).as_posix())
        os.replace(archive_tmp, destination)
    finally:
        if archive_tmp.exists():
            archive_tmp.unlink()
        shutil.rmtree(staging, ignore_errors=True)

    return {
        "path": str(destination),
        "stream_count": len(streams),
        "tensor_extra_count": len(tensor_entries),
        "skipped_extras": skipped,
    }


def load_h3_latent(source: str | Path) -> dict:
    """Load an H3 AV latent archive without executing arbitrary archive paths."""
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"H3 latent file does not exist: {source}")

    staging = Path(tempfile.mkdtemp(prefix=".mmxlatent-load-"))
    try:
        names = _extract_archive(source, staging)
        manifest_path = staging / "manifest.json"
        if "manifest.json" not in names:
            raise ValueError("H3 latent archive has no manifest.json.")
        manifest = _validate_manifest(_read_json(manifest_path, "manifest"), names)

        streams = [_load_tensor(staging / item["path"]) for item in manifest["streams"]]
        payload: dict[str, Any] = {"samples": _repack_av_streams(streams)}

        for item in manifest.get("tensor_extras", []):
            key = str(item.get("key", ""))
            if not key or key == "samples":
                continue
            payload[key] = _load_tensor(staging / item["path"])

        json_path = manifest.get("json_extras_path")
        if json_path:
            extras = _read_json(staging / json_path, "JSON extras")
            if isinstance(extras, dict):
                payload.update({str(key): value for key, value in extras.items() if key != "samples"})

        metadata_path = manifest.get("metadata_path")
        if metadata_path:
            metadata = _read_json(staging / metadata_path, "metadata")
            if isinstance(metadata, dict):
                payload["mmx_latent_metadata"] = metadata
        return payload
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def latent_store_root() -> Path:
    """Return the ComfyUI output subdirectory used by standalone latent nodes."""
    import folder_paths

    root = Path(folder_paths.get_output_directory()) / "minimax_h3_latents"
    root.mkdir(parents=True, exist_ok=True)
    return root


def list_latent_files() -> list[str]:
    root = latent_store_root()
    out = []
    for path in root.rglob(f"*{FILE_SUFFIX}"):
        if path.is_file():
            out.append(path.relative_to(root).as_posix())
    return sorted(out, key=str.casefold)


def resolve_latent_file(name: str) -> Path:
    root = latent_store_root().resolve()
    raw = str(name or "").replace("\\", "/")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("Latent filename must stay inside the H3 latent store.") from exc
    if not candidate.name.endswith(FILE_SUFFIX):
        raise ValueError(f"Latent filename must end with {FILE_SUFFIX}.")
    if not candidate.is_file():
        raise FileNotFoundError(f"Saved H3 latent not found: {name}")
    return candidate
