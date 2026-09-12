"""Release GPU memory between MiniMax H3 Director segment runs."""

from __future__ import annotations

import gc
import logging

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.vram")


def resolve_unload_models_policy(mode: str | None = None) -> tuple[bool, str]:
    """Resolve whether segment cleanup should unload models.

    ``auto`` keeps models resident on normal/high VRAM machines, where model
    reloads dominate multi-segment runs. Low/no/shared VRAM modes still use
    the conservative unload behavior. The explicit modes are useful when a
    workflow needs deterministic memory behavior.
    """
    requested = str(mode or "auto").strip().lower()
    if requested == "fast":
        return False, "fast"
    if requested == "stable":
        return True, "stable"
    if requested not in {"", "auto"}:
        log.warning("Unknown segment memory mode %r; using auto.", mode)

    try:
        import comfy.model_management as mm

        vram_state = str(getattr(getattr(mm, "vram_state", None), "name", ""))
        vram_name = vram_state.upper()
        if vram_name in {"NO_VRAM", "LOW_VRAM", "SHARED"}:
            return True, f"auto/{vram_state.lower()}"
        if vram_name in {"NORMAL_VRAM", "HIGH_VRAM", "DISABLED"}:
            return False, f"auto/{vram_state.lower()}"
        return True, "auto/safe"
    except Exception as exc:
        # Memory safety wins if ComfyUI changes its model-management API.
        log.debug("Could not inspect ComfyUI VRAM state: %s", exc)
        return True, "auto/safe"


def cleanup_segment_vram(*, enabled: bool = True, unload_models: bool = True) -> None:
    """Release segment GPU memory: gc, optional unload of ComfyUI models, empty CUDA cache."""
    if not enabled:
        return
    gc.collect()
    try:
        import comfy.model_management as mm

        mm.cleanup_models_gc()
        if unload_models:
            mm.unload_all_models()
            mm.cleanup_models()
        mm.soft_empty_cache()
    except Exception as exc:
        log.warning("Segment VRAM cleanup failed: %s", exc)
        return
    if unload_models:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (models unloaded, cache cleared)")
    else:
        log.debug("MiniMax H3 Director: segment VRAM cleanup (cache cleared, models kept loaded)")
