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
