"""Standalone MiniMax H3 latent workflow nodes.

These nodes intentionally do not depend on the Director timeline.  They are
small building blocks for workflows such as:

    H3 conditioning/first sampler -> Latent Refine -> VAE decode
                                      |
                                      +-> Latent Save / Latent Load
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import comfy.samplers

from ..director.core_sampling import sample_single_stage
from ..director.latent_file import (
    FILE_SUFFIX,
    latent_store_root,
    list_latent_files,
    load_h3_latent,
    resolve_latent_file,
    save_h3_latent,
)
from ..director.h3_motion_context import _streams_from_latent
from ..director.latent_queue import (
    allocate_queue_archive,
    latent_queue_root,
    list_latent_queues,
    load_queue_record,
    new_queue_record,
    select_queue_items,
)


def _safe_prefix(raw: str) -> Path:
    """Normalize a user prefix while keeping it under the latent store."""
    value = str(raw or "").strip().replace("\\", "/")
    if value.lower().endswith(FILE_SUFFIX):
        value = value[: -len(FILE_SUFFIX)]
    elif value.lower().endswith(".zip"):
        value = value[:-4]
    parts = []
    for part in value.split("/"):
        part = re.sub(r'[<>:"|?*\x00-\x1f]', "_", part).strip(" .")
        if not part or part in {".", ".."}:
            continue
        if part.startswith("."):
            part = "_" + part[1:]
        parts.append(part)
    return Path(*parts) if parts else Path("h3_latent")


def _next_latent_path(prefix: str, overwrite: bool) -> Path:
    root = latent_store_root()
    relative = _safe_prefix(prefix)
    candidate = root / f"{relative}{FILE_SUFFIX}"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    if overwrite or not candidate.exists():
        return candidate
    for index in range(1, 100000):
        stem = candidate.name[: -len(FILE_SUFFIX)]
        numbered = candidate.with_name(f"{stem}_{index:05d}{FILE_SUFFIX}")
        if not numbered.exists():
            return numbered
    raise RuntimeError("Could not allocate a free H3 latent filename.")


class MiniMaxH3LatentRefine:
    """Run an independent second sampling pass directly on an H3 AV LATENT."""

    CATEGORY = "MiniMaxH3/Latent"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "status")
    FUNCTION = "execute"

    DESCRIPTION = (
        "Direct MiniMax H3 second sampling pass. Connect the first sampler's "
        "H3 AV LATENT here; no VAE encode/decode or Director timeline is used. "
        "The output can go to another pass, VAE Decode, or Latent Save."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latent": ("LATENT",),
                "sampler": (list(comfy.samplers.KSampler.SAMPLERS), {"default": "euler"}),
                "scheduler": (
                    list(comfy.samplers.KSampler.SCHEDULERS),
                    {"default": "simple"},
                ),
                "steps": ("INT", {"default": 3, "min": 1, "max": 100}),
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 30.0, "step": 0.05}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {
                "negative": ("CONDITIONING",),
                "sigmas": ("SIGMAS", {"forceInput": True}),
                "denoise": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shift_video": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "shift_audio": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 100.0, "step": 0.1}),
                "apply_shift": ("BOOLEAN", {"default": True}),
            },
        }

    def execute(
        self,
        model,
        positive,
        latent,
        sampler,
        scheduler,
        steps,
        cfg,
        seed,
        negative=None,
        sigmas=None,
        denoise=0.85,
        shift_video=12.0,
        shift_audio=3.0,
        apply_shift=True,
    ):
        if not isinstance(latent, dict) or "samples" not in latent:
            raise ValueError("MiniMax H3 Latent Refine requires an H3 LATENT input.")
        if positive is None:
            raise ValueError("MiniMax H3 Latent Refine requires positive conditioning.")
        try:
            stream_count = len(_streams_from_latent(latent))
        except Exception as exc:
            raise ValueError(
                "MiniMax H3 Latent Refine requires the official H3 AV NestedTensor "
                "latent (video and audio streams)."
            ) from exc
        if stream_count < 2:
            raise ValueError(
                "MiniMax H3 Latent Refine requires both video and audio H3 latent streams."
            )

        negative_use = negative if negative else []
        cfg_use = float(cfg)
        basic_guider = not negative_use and abs(cfg_use - 1.0) < 1e-6
        if not negative_use and not basic_guider:
            # H3's no-negative workflow uses BasicGuider; an empty negative
            # conditioning cannot provide meaningful CFG guidance.
            cfg_use = 1.0

        sampled = sample_single_stage(
            model=model,
            positive=positive,
            negative=negative_use,
            latent=latent,
            seed=int(seed),
            cfg=cfg_use,
            steps=int(steps),
            sampler_name=str(sampler),
            scheduler=str(scheduler),
            shift_video=float(shift_video),
            shift_audio=float(shift_audio),
            denoise=float(denoise),
            sigmas=sigmas,
            apply_shift=True if apply_shift is None else bool(apply_shift),
        )
        status = (
            f"H3 latent refine complete: steps={int(steps)}, sampler={sampler}, "
            f"scheduler={scheduler}, cfg={cfg_use:g}"
        )
        if sigmas is not None:
            status += ", sigmas=external"
        if not negative_use and float(cfg) != 1.0:
            status += " (empty negative -> BasicGuider/cfg=1)"
        return (sampled, status)


class MiniMaxH3LatentSave:
    """Save an H3 AV latent into ``ComfyUI/output/minimax_h3_latents``."""

    CATEGORY = "MiniMaxH3/Latent"
    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("latent", "path")
    FUNCTION = "save"
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Save video/audio H3 latent streams as a portable .mmxlatent.zip file "
        "under the ComfyUI output/minimax_h3_latents directory, while passing "
        "the same LATENT through."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "filename_prefix": ("STRING", {"default": "h3_latent"}),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(self, latent, filename_prefix="h3_latent", overwrite=False, prompt=None, extra_pnginfo=None):
        destination = _next_latent_path(filename_prefix, bool(overwrite))
        metadata = {}
        if prompt is not None:
            metadata["prompt"] = prompt
        if extra_pnginfo is not None:
            metadata["extra_pnginfo"] = extra_pnginfo
        info = save_h3_latent(latent, destination, metadata=metadata)
        root = latent_store_root()
        relative = Path(info["path"]).relative_to(root).as_posix()
        return {
            "ui": {
                "latents": [{"filename": relative, "subfolder": "", "type": "output"}],
            },
            "result": (latent, str(destination)),
        }


class MiniMaxH3LatentLoad:
    """Load a previously saved H3 AV latent from the local latent store."""

    CATEGORY = "MiniMaxH3/Latent"
    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "load"

    DESCRIPTION = (
        "Load a .mmxlatent.zip file saved by MiniMax H3 Latent Save. "
        "Refresh the node/workflow after creating a new file so the filename "
        "list is rebuilt."
    )

    @classmethod
    def INPUT_TYPES(cls):
        files = list_latent_files()
        if not files:
            files = ["(no saved H3 latent files)"]
        return {"required": {"filename": (files,)}}

    @staticmethod
    def IS_CHANGED(filename):
        try:
            path = resolve_latent_file(filename)
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(block)
            return digest.hexdigest()
        except Exception:
            return f"missing:{filename}"

    def load(self, filename):
        if str(filename).startswith("("):
            raise ValueError("No saved H3 latent file is available yet.")
        return (load_h3_latent(resolve_latent_file(filename)),)


class MiniMaxH3LatentQueueSave:
    """Save one H3 latent together with the prompt that belongs to it."""

    CATEGORY = "MiniMaxH3/Latent Queue"
    RETURN_TYPES = ("LATENT", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("latent", "prompt", "record_id", "path")
    FUNCTION = "save"
    OUTPUT_NODE = True

    DESCRIPTION = (
        "Append one AV latent + prompt pair to an ordered H3 queue. "
        "The prompt record is embedded in the same archive as the latent."
    )

    @staticmethod
    def IS_CHANGED(**kwargs):
        del kwargs
        # Queue Save is an append operation. It must run again when the user
        # queues the same graph a second time, even if ComfyUI's intermediate
        # cache sees identical widget/link values.
        import time

        return time.time_ns()

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "latent": ("LATENT",),
                "prompt_text": (
                    "STRING",
                    {"multiline": True, "default": "", "tooltip": "Prompt paired with this latent."},
                ),
                "queue_name": (
                    "STRING",
                    {"default": "h3_refine_queue", "tooltip": "Folder name under output/minimax_h3_latents/queues."},
                ),
                "item_name": (
                    "STRING",
                    {"default": "shot", "tooltip": "Visible filename label, e.g. shot_01 or closeup_dog."},
                ),
                "queue_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999999, "tooltip": "0 = append; explicit number keeps a stable queue ID."},
                ),
                "overwrite": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "negative_prompt_text": ("STRING", {"multiline": True, "default": ""}),
                "task_key": ("STRING", {"default": "t2v"}),
                "width": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
                "height": ("INT", {"default": 0, "min": 0, "max": 8192, "step": 32}),
                "length": ("INT", {"default": 0, "min": 0, "max": 3600}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def save(
        self,
        latent,
        prompt_text="",
        queue_name="h3_refine_queue",
        item_name="shot",
        queue_index=0,
        overwrite=False,
        negative_prompt_text="",
        task_key="t2v",
        width=0,
        height=0,
        length=0,
        prompt=None,
        extra_pnginfo=None,
    ):
        destination, index, record_id = allocate_queue_archive(
            queue_name,
            item_name,
            queue_index=int(queue_index),
            overwrite=bool(overwrite),
        )
        record = new_queue_record(
            queue_name=queue_name,
            queue_index=index,
            record_id=record_id,
            item_name=item_name,
            prompt=prompt_text,
            negative_prompt=negative_prompt_text,
            task_key=task_key,
            width=width,
            height=height,
            length=length,
        )
        metadata = {
            "queue_record": record,
            "queue_prompt": str(prompt_text or ""),
        }
        # Keep the original Comfy graph as optional audit metadata, but never
        # use it for pairing.  The explicit queue_record is authoritative.
        if prompt is not None:
            metadata["comfy_prompt"] = prompt
        if extra_pnginfo is not None:
            metadata["extra_pnginfo"] = extra_pnginfo
        info = save_h3_latent(latent, destination, metadata=metadata)
        relative = destination.relative_to(latent_queue_root()).as_posix()
        status = (
            f"H3 queue saved: {queue_name} / #{index:04d} / {record_id}"
        )
        return {
            "ui": {
                "latents": [{"filename": relative, "subfolder": "", "type": "output"}],
                "queue": [status],
            },
            "result": (latent, str(prompt_text or ""), record_id, str(destination)),
        }


class MiniMaxH3LatentQueueLoad:
    """Read paired H3 latent/prompt records, optionally advancing a cursor."""

    CATEGORY = "MiniMaxH3/Latent Queue"
    RETURN_TYPES = ("LATENT", "STRING", "STRING", "STRING", "STRING", "INT", "STRING")
    RETURN_NAMES = ("latent", "prompt", "negative_prompt", "record_id", "item_name", "remaining", "status")
    OUTPUT_IS_LIST = (True, True, True, True, True, True, True)
    FUNCTION = "load"

    DESCRIPTION = (
        "Read an ordered batch of H3 latent + prompt pairs. "
        "next mode advances one Comfy node cursor; index mode is deterministic."
    )

    @classmethod
    def INPUT_TYPES(cls):
        queues = list_latent_queues()
        if not queues:
            queues = ["(no H3 latent queues)"]
        return {
            "required": {
                "queue_name": (queues,),
                "read_mode": (["next", "index"], {"default": "next"}),
                "queue_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999999, "tooltip": "index mode: first queue ID; next mode: initial ID when cursor is new."},
                ),
                "batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 128, "tooltip": "Number of paired records read this execution; downstream nodes map them one by one."},
                ),
                "reset_token": (
                    "INT",
                    {"default": 0, "min": 0, "max": 9999999, "tooltip": "Increase this number to restart next mode from queue_index."},
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, queue_name, read_mode="next", queue_index=0, batch_size=1, reset_token=0, **kwargs):
        del queue_name, queue_index, batch_size, reset_token, kwargs
        # A next-mode reader is intentionally live: each Queue Prompt consumes
        # the next records instead of returning the previous cached list.
        if str(read_mode).lower() == "next":
            import time

            return time.time_ns()
        return f"index:{read_mode}"

    def load(
        self,
        queue_name,
        read_mode="next",
        queue_index=0,
        batch_size=1,
        reset_token=0,
        unique_id=None,
    ):
        if str(queue_name).startswith("("):
            raise ValueError("No H3 latent queue is available yet.")
        paths, remaining_after = select_queue_items(
            queue_name,
            read_mode=read_mode,
            queue_index=int(queue_index),
            batch_size=int(batch_size),
            reset_token=int(reset_token),
            unique_id=unique_id,
        )
        latents = []
        prompts = []
        negatives = []
        record_ids = []
        item_names = []
        remaining = []
        statuses = []
        for position, path in enumerate(paths):
            latent, record = load_queue_record(path)
            record_id = str(record.get("record_id") or path.stem)
            latents.append(latent)
            prompts.append(str(record.get("prompt") or ""))
            negatives.append(str(record.get("negative_prompt") or ""))
            record_ids.append(record_id)
            item_names.append(str(record.get("item_name") or record_id))
            remaining.append(max(0, int(remaining_after) + len(paths) - position - 1))
            statuses.append(
                f"H3 queue read: {queue_name} / {record_id} / prompt paired"
            )
        return latents, prompts, negatives, record_ids, item_names, remaining, statuses
