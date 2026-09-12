"""Portable MiniMax H3 AV-latent archives.

The Director already keeps CPU copies of per-segment AV latents for
continuity and ``confirm_first_pass``.  This module turns those internal
cache files into a small, portable archive without exposing PyTorch pickle
files to imports.  Tensor payloads use safetensors; the legacy ``.av.pt``
files remain an internal compatibility format only.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import re
import shutil
import tempfile
import time
import uuid
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any

import torch

import folder_paths

from .h3_motion_context import _repack_av_streams, _streams_from_latent
from .segment_cache import _safe_unlink, _write_via_temp

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.latent_pack")

LATENT_FORMAT = "minimax-h3-director-latent"
LATENT_VERSION = 1
LATENT_SUFFIX = ".mmxlatent.zip"
LATENT_EXPORT_TTL_SEC = 60 * 60
LATENT_EXPORT_INLINE_MAX = 48 * 1024 * 1024
LATENT_UPLOAD_MAX = 8 * 1024 * 1024 * 1024
LATENT_UNCOMPRESSED_MAX = 16 * 1024 * 1024 * 1024
LATENT_MAX_ENTRIES = 16000
LATENT_MAX_SINGLE_FILE = 8 * 1024 * 1024 * 1024
LATENT_ZIP_CHUNK = 1024 * 1024
ASCII_PATH_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
NODE_ID_RE = re.compile(r"^\d+$")
SEGMENT_FILE_RE = re.compile(r"^seg_(\d+)(\.pre)?\.av\.pt$")


def _require_node_id(node_id: str | None) -> str:
    value = str(node_id or "").strip()
    if not NODE_ID_RE.fullmatch(value):
        raise ValueError("Invalid Director node id.")
    return value


def _cache_root_for(node_id: str, *, create: bool = False) -> Path:
    root = Path(folder_paths.get_output_directory()) / "minimax_seg_cache" / node_id
    if create:
        root.mkdir(parents=True, exist_ok=True)
    return root


def _export_root() -> Path:
    root = Path(folder_paths.get_temp_directory()) / "minimax_director_latent_export"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _purge_exports(*, keep: str | None = None) -> None:
    try:
        now = time.time()
        for path in _export_root().glob(f"*{LATENT_SUFFIX}"):
            if keep and path.name == keep:
                continue
            try:
                if now - path.stat().st_mtime >= LATENT_EXPORT_TTL_SEC:
                    _safe_unlink(path)
            except OSError:
                continue
    except OSError:
        return


def _safetensor_io():
    try:
        from safetensors.torch import load_file, save_file
    except ImportError as exc:
        raise RuntimeError(
            "Latent 包需要 safetensors。请在 ComfyUI 的虚拟环境中执行 "
            "pip install safetensors 后重启。"
        ) from exc
    return load_file, save_file


def _json_safe(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, float, str)):
        return True
    if isinstance(value, (list, tuple)):
        return all(_json_safe(item) for item in value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _json_safe(item) for key, item in value.items())
    return False


def _latent_streams(payload: Any) -> list[torch.Tensor]:
    if not isinstance(payload, dict) or "samples" not in payload:
        raise ValueError("AV latent payload is missing samples.")
    try:
        streams = list(_streams_from_latent(payload))
    except Exception as exc:
        raise ValueError(f"Invalid MiniMax H3 AV latent: {exc}") from exc
    if not streams or any(not torch.is_tensor(item) for item in streams):
        raise ValueError("AV latent streams must be tensors.")
    return [item.detach().cpu().contiguous() for item in streams]


def _save_tensor(path: Path, tensor: torch.Tensor) -> None:
    _load_file, save_file = _safetensor_io()
    del _load_file
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({"tensor": tensor.detach().cpu().contiguous()}, str(path))


def _load_tensor(path: Path) -> torch.Tensor:
    load_file, _save_file = _safetensor_io()
    loaded = load_file(str(path), device="cpu")
    if set(loaded) != {"tensor"} or not torch.is_tensor(loaded.get("tensor")):
        raise ValueError(f"Invalid tensor artifact: {path.name}")
    return loaded["tensor"].contiguous()


def _artifact_prefix(index: int, cache_kind: str) -> str:
    return f"segments/{int(index):04d}/{cache_kind}"


def _read_json_object(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.warning("Ignoring invalid latent cache metadata %s: %s", path, exc)
        return None
    return value if isinstance(value, dict) else None


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _export_one_artifact(
    *,
    source: Path,
    staging: Path,
    segment_index: int,
    cache_kind: str,
    warnings: list[str],
) -> dict[str, Any] | None:
    try:
        payload = torch.load(source, map_location="cpu", weights_only=False)
        streams = _latent_streams(payload)
    except Exception as exc:
        warnings.append(f"片段 {segment_index + 1} 的 {cache_kind} latent 无法读取：{exc}")
        return None

    prefix = _artifact_prefix(segment_index, cache_kind)
    artifact: dict[str, Any] = {
        "segment_index": int(segment_index),
        "cache_kind": cache_kind,
        "stream_count": len(streams),
        "video": f"{prefix}/video.safetensors",
        "video_shape": [int(x) for x in streams[0].shape],
        "video_dtype": str(streams[0].dtype).replace("torch.", ""),
    }
    _save_tensor(staging / artifact["video"], streams[0])

    if len(streams) > 1:
        artifact["audio"] = f"{prefix}/audio.safetensors"
        artifact["audio_shape"] = [int(x) for x in streams[1].shape]
        artifact["audio_dtype"] = str(streams[1].dtype).replace("torch.", "")
        _save_tensor(staging / artifact["audio"], streams[1])
    else:
        warnings.append(f"片段 {segment_index + 1} 的 {cache_kind} latent 没有音频流。")

    extras: list[dict[str, str]] = []
    extra_values: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "samples":
            continue
        if torch.is_tensor(value):
            rel = f"{prefix}/extra_{len(extras):02d}.safetensors"
            _save_tensor(staging / rel, value)
            extras.append({"key": str(key), "path": rel})
        elif _json_safe(value):
            extra_values[str(key)] = value
        else:
            warnings.append(
                f"片段 {segment_index + 1} 的 {cache_kind} latent 附加字段 {key!r} 未导出。"
            )
    if extras:
        artifact["tensor_extras"] = extras
    if extra_values:
        rel = f"{prefix}/extra.json"
        _write_json(staging / rel, extra_values)
        artifact["json_extras"] = rel

    root = source.parent
    stem = f"seg_{segment_index:04d}"
    is_first = cache_kind == "first_pass"
    suffix = ".pre" if is_first else ""
    meta = _read_json_object(root / f"{stem}{suffix}.meta.json")
    handoff = _read_json_object(root / f"{stem}{suffix}.handoff.json")
    if meta is not None:
        rel = f"{prefix}/meta.json"
        _write_json(staging / rel, meta)
        artifact["meta"] = rel
    if handoff is not None:
        rel = f"{prefix}/handoff.json"
        _write_json(staging / rel, handoff)
        artifact["handoff"] = rel
    return artifact


def build_export_latent_pack(node_id: str | None) -> dict[str, Any]:
    """Build a portable latent archive from the Director's internal cache."""
    node = _require_node_id(node_id)
    root = _cache_root_for(node)
    if not root.is_dir():
        raise ValueError("当前 Director 没有分段 latent 缓存。先运行一次生成或确认一采。")

    files: list[tuple[int, str, Path]] = []
    for path in root.glob("seg_*.av.pt"):
        match = SEGMENT_FILE_RE.fullmatch(path.name)
        if not match:
            continue
        index = int(match.group(1))
        cache_kind = "first_pass" if match.group(2) else "final"
        files.append((index, cache_kind, path))
    files.sort(key=lambda row: (row[0], 0 if row[1] == "first_pass" else 1))
    if not files:
        raise ValueError("当前 Director 没有可导出的 AV latent 缓存。")

    staging = Path(tempfile.mkdtemp(prefix="mmx_latent_build_"))
    warnings: list[str] = []
    artifacts: list[dict[str, Any]] = []
    try:
        for index, cache_kind, source in files:
            artifact = _export_one_artifact(
                source=source,
                staging=staging,
                segment_index=index,
                cache_kind=cache_kind,
                warnings=warnings,
            )
            if artifact is not None:
                artifacts.append(artifact)
        if not artifacts:
            raise ValueError("缓存文件存在，但没有可读取的 AV latent。")

        manifest = {
            "format": LATENT_FORMAT,
            "version": LATENT_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
            "source": "MiniMax H3 Director segment cache",
            "segments": artifacts,
            "counts": {
                "artifacts": len(artifacts),
                "final": sum(a["cache_kind"] == "final" for a in artifacts),
                "first_pass": sum(a["cache_kind"] == "first_pass" for a in artifacts),
            },
        }
        _write_json(staging / "manifest.json", manifest)

        stored_name = f"{uuid.uuid4().hex}{LATENT_SUFFIX}"
        _purge_exports(keep=stored_name)
        zip_path = _export_root() / stored_name
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in staging.rglob("*"):
                if not path.is_file():
                    continue
                rel = str(path.relative_to(staging)).replace(os.sep, "/")
                if not _safe_archive_path(rel):
                    raise ValueError(f"Invalid latent archive path: {rel}")
                archive.write(path, rel)
        size = int(zip_path.stat().st_size)
        if size <= 0:
            _safe_unlink(zip_path)
            raise ValueError("Latent archive is empty.")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        return {
            "filename": stored_name,
            "downloadName": f"MiniMaxH3Director-Latent-{stamp}{LATENT_SUFFIX}",
            "bytes": size,
            "segments": artifacts,
            "counts": manifest["counts"],
            "warnings": warnings,
        }
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _safe_archive_path(name: str) -> bool:
    rel = str(name or "").replace("\\", "/").strip()
    if not rel or rel.startswith("/") or rel.endswith("/"):
        return False
    parts = [part for part in rel.split("/") if part]
    if any(part in {".", ".."} or part.startswith(".") for part in parts):
        return False
    return bool(ASCII_PATH_RE.fullmatch(rel))


def _validate_zip_entry(info: zipfile.ZipInfo, total: int) -> tuple[str, int]:
    rel = info.filename.replace("\\", "/").strip()
    if not rel or rel.endswith("/"):
        return "", total
    if not _safe_archive_path(rel):
        raise ValueError(f"Unsafe path in latent archive: {info.filename}")
    mode = (int(info.external_attr) >> 16) & 0o170000
    if mode == 0o120000:
        raise ValueError("Latent archive contains a symlink.")
    size = max(0, int(info.file_size or 0))
    if size > LATENT_MAX_SINGLE_FILE:
        raise ValueError("A file in the latent archive exceeds the size limit.")
    total += size
    if total > LATENT_UNCOMPRESSED_MAX:
        raise ValueError("Latent archive uncompressed size exceeds the limit.")
    return rel, total


def _extract_latent_zip(zip_path: Path, dest: Path) -> set[str]:
    try:
        if zip_path.stat().st_size > LATENT_UPLOAD_MAX:
            raise ValueError("Latent archive exceeds the upload size limit.")
    except OSError as exc:
        raise ValueError("Latent archive was not found.") from exc
    dest.mkdir(parents=True, exist_ok=True)
    extracted: set[str] = set()
    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()
        if len(infos) > LATENT_MAX_ENTRIES:
            raise ValueError("Latent archive contains too many files.")
        total = 0
        for info in infos:
            rel, total = _validate_zip_entry(info, total)
            if not rel:
                continue
            if rel in extracted:
                raise ValueError(f"Latent archive contains a duplicate path: {rel}")
            target = dest / rel.replace("/", os.sep)
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(info) as source, target.open("wb") as out:
                shutil.copyfileobj(source, out, length=LATENT_ZIP_CHUNK)
            extracted.add(rel)
    if "manifest.json" not in extracted:
        raise ValueError("Latent archive is missing manifest.json.")
    return extracted


def _artifact_path(manifest: dict[str, Any], artifact: dict[str, Any], key: str) -> str | None:
    value = artifact.get(key)
    if value is None:
        return None
    if not isinstance(value, str) or not _safe_archive_path(value):
        raise ValueError(f"Invalid latent archive artifact path: {key}")
    index = int(artifact.get("segment_index", -1))
    kind = str(artifact.get("cache_kind") or "")
    if index < 0 or kind not in {"first_pass", "final"}:
        raise ValueError("Invalid latent archive segment metadata.")
    prefix = _artifact_prefix(index, kind)
    if key == "video" and value != f"{prefix}/video.safetensors":
        raise ValueError("Latent archive video path does not match its segment.")
    if key == "audio" and value != f"{prefix}/audio.safetensors":
        raise ValueError("Latent archive audio path does not match its segment.")
    if key == "meta" and value != f"{prefix}/meta.json":
        raise ValueError("Latent archive metadata path does not match its segment.")
    if key == "handoff" and value != f"{prefix}/handoff.json":
        raise ValueError("Latent archive handoff path does not match its segment.")
    return value


def _validate_manifest(manifest: Any, extracted: set[str]) -> list[dict[str, Any]]:
    if not isinstance(manifest, dict):
        raise ValueError("Invalid latent archive manifest.")
    if manifest.get("format") != LATENT_FORMAT or int(manifest.get("version") or 0) != LATENT_VERSION:
        raise ValueError("Unsupported MiniMax H3 latent archive format/version.")
    raw_segments = manifest.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        raise ValueError("Latent archive contains no segments.")
    if len(raw_segments) > LATENT_MAX_ENTRIES:
        raise ValueError("Latent archive contains too many latent artifacts.")
    artifacts: list[dict[str, Any]] = []
    seen: set[tuple[int, str]] = set()
    for raw in raw_segments:
        if not isinstance(raw, dict):
            raise ValueError("Invalid latent archive segment entry.")
        try:
            index = int(raw.get("segment_index"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid latent archive segment index.") from exc
        kind = str(raw.get("cache_kind") or "")
        if index < 0 or index > 999999 or kind not in {"first_pass", "final"}:
            raise ValueError("Invalid latent archive segment metadata.")
        identity = (index, kind)
        if identity in seen:
            raise ValueError("Latent archive contains duplicate segment artifacts.")
        seen.add(identity)
        artifact = dict(raw)
        video_rel = _artifact_path(manifest, artifact, "video")
        assert video_rel is not None
        if video_rel not in extracted:
            raise ValueError(f"Latent archive is missing {video_rel}.")
        for key in ("audio", "meta", "handoff"):
            rel = _artifact_path(manifest, artifact, key)
            if rel is not None and rel not in extracted:
                raise ValueError(f"Latent archive is missing {rel}.")
        for item in artifact.get("tensor_extras") or []:
            if not isinstance(item, dict) or not isinstance(item.get("key"), str):
                raise ValueError("Invalid latent tensor extra metadata.")
            rel = item.get("path")
            if not isinstance(rel, str) or not _safe_archive_path(rel) or rel not in extracted:
                raise ValueError("Latent tensor extra is missing or unsafe.")
            if not rel.startswith(f"{_artifact_prefix(index, kind)}/extra_") or not rel.endswith(".safetensors"):
                raise ValueError("Latent tensor extra path does not match its segment.")
            if item["key"] == "samples":
                raise ValueError("Latent tensor extra cannot be named samples.")
        json_rel = artifact.get("json_extras")
        if json_rel is not None:
            if not isinstance(json_rel, str) or json_rel != f"{_artifact_prefix(index, kind)}/extra.json" or json_rel not in extracted:
                raise ValueError("Latent JSON extras are missing or unsafe.")
        artifacts.append(artifact)
    return artifacts


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_import_artifact(extracted: Path, artifact: dict[str, Any]) -> dict[str, Any]:
    video_rel = _artifact_path({}, artifact, "video")
    assert video_rel is not None
    streams = [_load_tensor(extracted / video_rel)]
    audio_rel = _artifact_path({}, artifact, "audio")
    if audio_rel is not None:
        streams.append(_load_tensor(extracted / audio_rel))
    payload: dict[str, Any] = {"samples": _repack_av_streams(streams)}
    tensor_extras = artifact.get("tensor_extras") or []
    if not isinstance(tensor_extras, list):
        raise ValueError("Invalid latent tensor extras.")
    for item in tensor_extras:
        key = str(item["key"])
        if key in payload:
            raise ValueError(f"Duplicate latent extra key: {key}")
        payload[key] = _load_tensor(extracted / str(item["path"]))
    json_rel = artifact.get("json_extras")
    if json_rel:
        extras = _load_json(extracted / str(json_rel))
        if not isinstance(extras, dict):
            raise ValueError("Invalid latent JSON extras.")
        for key, value in extras.items():
            if key == "samples" or key in payload:
                raise ValueError(f"Duplicate latent extra key: {key}")
            if not _json_safe(value):
                raise ValueError(f"Unsafe latent extra value: {key}")
            payload[key] = value
    return payload


def _copy_json_artifact(extracted: Path, rel: str | None, dest: Path) -> None:
    if rel is None:
        return
    value = _load_json(extracted / rel)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid latent metadata: {Path(rel).name}")
    _write_via_temp(dest, lambda path: _write_json(path, value))


def import_latent_pack(zip_path: Path, node_id: str | None) -> dict[str, Any]:
    """Validate and import a portable archive into a Director cache."""
    node = _require_node_id(node_id)
    extracted = Path(tempfile.mkdtemp(prefix="mmx_latent_import_"))
    try:
        names = _extract_latent_zip(zip_path, extracted)
        manifest = _load_json(extracted / "manifest.json")
        artifacts = _validate_manifest(manifest, names)
        loaded = [(artifact, _load_import_artifact(extracted, artifact)) for artifact in artifacts]

        root = _cache_root_for(node, create=True)
        imported = {"final": 0, "first_pass": 0}
        for artifact, payload in loaded:
            index = int(artifact["segment_index"])
            kind = str(artifact["cache_kind"])
            stem = f"seg_{index:04d}"
            suffix = ".pre" if kind == "first_pass" else ""
            latent_dest = root / f"{stem}{suffix}.av.pt"
            _write_via_temp(latent_dest, lambda path, data=payload: torch.save(data, path))
            _copy_json_artifact(
                extracted,
                _artifact_path(manifest, artifact, "meta"),
                root / f"{stem}{suffix}.meta.json",
            )
            if _artifact_path(manifest, artifact, "meta") is None:
                _safe_unlink(root / f"{stem}{suffix}.meta.json")
            _copy_json_artifact(
                extracted,
                _artifact_path(manifest, artifact, "handoff"),
                root / f"{stem}{suffix}.handoff.json",
            )
            if _artifact_path(manifest, artifact, "handoff") is None:
                _safe_unlink(root / f"{stem}{suffix}.handoff.json")
            # Decoded frames are a different representation and may belong to
            # another latent/model. Force a fresh decode instead of silently
            # pairing old pixels with the imported tensor.
            _safe_unlink(root / f"{stem}{suffix}.pt")
            if kind == "final":
                _safe_unlink(root / f"{stem}.audio.pt")
            imported[kind] += 1
        return {
            "format": LATENT_FORMAT,
            "imported": imported,
            "artifacts": len(loaded),
            "segments": sorted({int(a["segment_index"]) + 1 for a, _ in loaded}),
        }
    finally:
        shutil.rmtree(extracted, ignore_errors=True)


def _resolve_uploaded_latent(name: str, subfolder: str = "", type_name: str = "input") -> Path:
    from .pack import resolve_media_path

    path = resolve_media_path(name, subfolder=subfolder, type_name=type_name)
    if path is None or not path.is_file() or path.suffix.lower() != ".zip":
        raise ValueError("Uploaded latent archive was not found or is not a .zip file.")
    return path


async def minimax_export_latents(request):
    try:
        body = await request.json()
    except Exception as exc:
        return _http_response(400, f"Invalid JSON: {exc}")
    try:
        result = build_export_latent_pack(body.get("node_id"))
        filename = str(result["filename"])
        path = _export_root() / filename
        download_name = re.sub(r"[^A-Za-z0-9._-]+", "_", str(result["downloadName"])) or filename
        if int(path.stat().st_size) <= 0:
            raise ValueError("Latent archive is empty.")
        if path.stat().st_size <= LATENT_EXPORT_INLINE_MAX:
            data = path.read_bytes()
            _safe_unlink(path)
            result["zipB64"] = base64.b64encode(data).decode("ascii")
            result["downloadName"] = download_name
            return _json_response(result)
        from .pack import _send_zip_file

        extra = {
            "X-Latent-Download-Name": download_name,
            "X-Latent-Counts": json.dumps(result.get("counts") or {}, ensure_ascii=True),
            "X-Latent-Warnings": json.dumps(result.get("warnings") or [], ensure_ascii=True),
        }
        return await _send_zip_file(request, path, download_name, extra, unlink_after=True)
    except Exception as exc:
        log.warning("Director latent export failed: %s", exc)
        return _http_response(400, str(exc))


async def minimax_import_latents(request):
    upload_dir: Path | None = None
    input_zip: Path | None = None
    try:
        ctype = request.content_type or ""
        if "multipart" in ctype:
            post = await request.post()
            node_id = str(post.get("node_id") or "").strip()
            upload = post.get("latent_pack") or post.get("pack")
            if upload is None or not hasattr(upload, "file"):
                return _http_response(400, "Missing latent_pack file.")
            upload_dir = Path(tempfile.mkdtemp(prefix="mmx_latent_upload_"))
            zip_path = upload_dir / "latent.zip"
            with zip_path.open("wb") as out:
                shutil.copyfileobj(upload.file, out, length=LATENT_ZIP_CHUNK)
        else:
            body = await request.json()
            node_id = str(body.get("node_id") or "").strip()
            zip_path = _resolve_uploaded_latent(
                str(body.get("filename") or body.get("name") or ""),
                subfolder=str(body.get("subfolder") or ""),
                type_name=str(body.get("type") or "input"),
            )
            try:
                input_root = Path(folder_paths.get_input_directory()).resolve()
                if zip_path.resolve().is_relative_to(input_root):
                    input_zip = zip_path
            except (OSError, ValueError, AttributeError):
                input_zip = None
        return _json_response(import_latent_pack(zip_path, node_id))
    except Exception as exc:
        log.warning("Director latent import failed: %s", exc)
        return _http_response(400, str(exc))
    finally:
        if upload_dir is not None:
            shutil.rmtree(upload_dir, ignore_errors=True)
        if input_zip is not None:
            _safe_unlink(input_zip)


def _json_response(data: dict[str, Any]):
    from aiohttp import web

    return web.json_response(data)


def _http_response(status: int, text: str):
    from aiohttp import web

    return web.Response(status=int(status), text=str(text))
