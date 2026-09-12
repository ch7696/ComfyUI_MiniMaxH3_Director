"""Ordered H3 latent + prompt queue storage.

The queue deliberately stores one prompt record inside the same archive as its
AV latent.  The filename is only a human-friendly index; the loader uses the
embedded record as the source of truth so a renamed file cannot pair a latent
with another prompt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .latent_file import FILE_SUFFIX, latent_store_root, load_h3_latent, save_h3_latent

QUEUE_DIR_NAME = "queues"
_QUEUE_INDEX_RE = re.compile(r"^(?P<index>[0-9]+)__.*__latent\.mmxlatent\.zip$", re.IGNORECASE)
_MAX_QUEUE_COMPONENT = 80


def latent_queue_root() -> Path:
    """Return ``ComfyUI/output/minimax_h3_latents/queues``."""
    root = latent_store_root() / QUEUE_DIR_NAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _safe_relative(raw: str, *, default: str) -> Path:
    value = str(raw or "").strip().replace("\\", "/")
    parts: list[str] = []
    for part in value.split("/"):
        part = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part).strip(" .")
        if not part or part in {".", ".."}:
            continue
        if part.startswith("."):
            part = "_" + part[1:]
        parts.append(part[:_MAX_QUEUE_COMPONENT])
    return Path(*parts) if parts else Path(default)


def _resolve_queue_dir(queue_name: str, *, must_exist: bool = False) -> Path:
    root = latent_queue_root().resolve()
    relative = _safe_relative(queue_name, default="h3_refine_queue")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError("H3 queue name must stay inside the queue store.") from exc
    if must_exist and not candidate.is_dir():
        raise FileNotFoundError(f"H3 latent queue does not exist: {queue_name}")
    return candidate


def _queue_file_index(path: Path) -> int:
    match = _QUEUE_INDEX_RE.match(path.name)
    if not match:
        return 2**31 - 1
    try:
        return int(match.group("index"))
    except ValueError:
        return 2**31 - 1


def list_latent_queues() -> list[str]:
    """List queue names that contain at least one queue archive."""
    root = latent_queue_root()
    names: set[str] = set()
    for path in root.rglob(f"*{FILE_SUFFIX}"):
        if not path.is_file() or any(part.startswith(".") for part in path.relative_to(root).parts):
            continue
        names.add(path.parent.relative_to(root).as_posix())
    return sorted(names, key=str.casefold)


def list_queue_items(queue_name: str) -> list[Path]:
    """Return queue archives in deterministic numeric/name order."""
    queue_dir = _resolve_queue_dir(queue_name, must_exist=True)
    items = [
        path
        for path in queue_dir.iterdir()
        if path.is_file() and path.name.lower().endswith(FILE_SUFFIX.lower())
    ]
    return sorted(items, key=lambda path: (_queue_file_index(path), path.name.casefold()))


def _slug(raw: str, *, default: str = "shot") -> str:
    value = str(raw or "").strip().replace("\\", "_").replace("/", "_")
    value = re.sub(r'[<>:"|?*\x00-\x1f]', "_", value)
    value = re.sub(r"\s+", "_", value).strip(" ._")
    if not value:
        value = default
    if value.startswith("."):
        value = "_" + value[1:]
    return value[:_MAX_QUEUE_COMPONENT] or default


def _record_id(path: Path) -> str:
    if path.name.lower().endswith(FILE_SUFFIX.lower()):
        return path.name[: -len(FILE_SUFFIX)]
    return path.stem


def _allocate_index(items: list[Path]) -> int:
    indexes = [_queue_file_index(path) for path in items]
    indexes = [index for index in indexes if index < 2**31 - 1]
    return max(indexes, default=0) + 1


def allocate_queue_archive(
    queue_name: str,
    item_name: str,
    *,
    queue_index: int = 0,
    overwrite: bool = False,
) -> tuple[Path, int, str]:
    """Allocate a visible archive path and return ``(path, index, record_id)``.

    ``queue_index=0`` appends after the current largest index.  An explicit
    index is stable and refuses accidental overwrite unless requested.
    """
    queue_dir = _resolve_queue_dir(queue_name)
    queue_dir.mkdir(parents=True, exist_ok=True)
    items = list_queue_items(queue_name) if queue_dir.exists() else []
    index = int(queue_index)
    if index <= 0:
        index = _allocate_index(items)
    slug = _slug(item_name)
    filename = f"{index:04d}__{slug}__latent{FILE_SUFFIX}"
    destination = queue_dir / filename
    if destination.exists() and not overwrite:
        raise FileExistsError(
            f"H3 queue item already exists: {destination.name}; "
            "choose another queue_index/item_name or enable overwrite."
        )
    return destination, index, _record_id(destination)


def _state_path(queue_name: str, unique_id: str | None) -> Path:
    root = latent_store_root() / ".queue_state"
    root.mkdir(parents=True, exist_ok=True)
    token = f"{queue_name}\0{unique_id or 'default'}".encode("utf-8", "replace")
    digest = hashlib.sha256(token).hexdigest()[:32]
    return root / f"{digest}.json"


def _read_state(queue_name: str, unique_id: str | None) -> dict:
    path = _state_path(queue_name, unique_id)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    return value if isinstance(value, dict) else {}


def _write_state(queue_name: str, unique_id: str | None, state: dict) -> None:
    path = _state_path(queue_name, unique_id)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(state, ensure_ascii=False, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initial_position(items: list[Path], queue_index: int) -> int:
    requested = int(queue_index)
    if requested <= 0:
        return 0
    for position, path in enumerate(items):
        if _queue_file_index(path) == requested:
            return position
    # A friendly fallback for queues imported from an external tool whose
    # filenames do not carry numeric IDs: queue_index is also 1-based position.
    return max(0, min(len(items), requested - 1))


def select_queue_items(
    queue_name: str,
    *,
    read_mode: str = "next",
    queue_index: int = 0,
    batch_size: int = 1,
    reset_token: int = 0,
    unique_id: str | None = None,
) -> tuple[list[Path], int]:
    """Select a batch and return ``(paths, remaining_after_batch)``.

    In ``next`` mode the cursor is persisted per queue + Comfy node ID.  In
    ``index`` mode no cursor is touched and ``queue_index`` is the first item
    ID (1-based fallback position).
    """
    items = list_queue_items(queue_name)
    if not items:
        raise FileNotFoundError(f"H3 latent queue is empty: {queue_name}")
    count = max(1, min(int(batch_size), 128))
    if str(read_mode or "next").lower() == "index":
        start = _initial_position(items, int(queue_index))
        if start >= len(items):
            raise IndexError(f"H3 queue index is past the end: {queue_index}")
        selected = items[start : start + count]
        return selected, max(0, len(items) - (start + len(selected)))

    state = _read_state(queue_name, unique_id)
    token = int(reset_token)
    if state.get("reset_token") != token:
        start = _initial_position(items, int(queue_index))
    else:
        try:
            start = int(state.get("cursor", 0))
        except (TypeError, ValueError):
            start = 0
        start = max(0, min(start, len(items)))
    if start >= len(items):
        raise IndexError(
            f"H3 latent queue is complete: {queue_name}. "
            "Increase reset_token to start again or use index mode."
        )
    selected = items[start : start + count]
    _write_state(
        queue_name,
        unique_id,
        {
            "queue": str(queue_name),
            "cursor": start + len(selected),
            "reset_token": token,
            "last_record_id": _record_id(selected[-1]),
            "updated_at": time.time(),
        },
    )
    return selected, max(0, len(items) - (start + len(selected)))


def load_queue_record(path: str | Path) -> tuple[dict, dict]:
    """Load an archive and return ``(latent, queue_record)``."""
    latent = load_h3_latent(path)
    metadata = latent.get("mmx_latent_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
    record = metadata.get("queue_record")
    if not isinstance(record, dict):
        # Accept a simple metadata shape for forward/backward compatibility.
        record = metadata
    record = {str(key): value for key, value in record.items()}
    record.setdefault("record_id", _record_id(Path(path)))
    record.setdefault("filename", Path(path).name)
    record.setdefault("prompt", "")
    record.setdefault("negative_prompt", "")
    return latent, record


def new_queue_record(
    *,
    queue_name: str,
    queue_index: int,
    record_id: str,
    item_name: str,
    prompt: str,
    negative_prompt: str = "",
    task_key: str = "t2v",
    width: int = 0,
    height: int = 0,
    length: int = 0,
) -> dict:
    return {
        "queue_name": str(queue_name),
        "queue_index": int(queue_index),
        "record_id": str(record_id),
        "item_name": str(item_name or "shot"),
        "prompt": str(prompt or ""),
        "negative_prompt": str(negative_prompt or ""),
        "task_key": str(task_key or "t2v"),
        "width": int(width or 0),
        "height": int(height or 0),
        "length": int(length or 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


__all__ = [
    "allocate_queue_archive",
    "latent_queue_root",
    "list_latent_queues",
    "list_queue_items",
    "load_queue_record",
    "new_queue_record",
    "select_queue_items",
]
