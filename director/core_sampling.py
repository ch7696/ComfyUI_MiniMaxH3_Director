"""Single-stage sampling for MiniMax H3 (SigmaShift + KSampler).

Modified by the MiniMax H3 Director T8 Bridge contributors in 2026 to add the
optional GPL-3.0-or-later T8 dual-clock sampling backend.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types
from functools import lru_cache
from pathlib import Path
from typing import Callable

log = logging.getLogger("ComfyUI-MiniMaxH3-Director.director.core_sampling")

PhaseCallback = Callable[[str, float], None]


@lru_cache(maxsize=1)
def _load_t8_sampling_module():
    """Load T8's sampler without relying on its hyphenated directory as an import name."""
    package_root = Path(__file__).resolve().parents[2] / "minimax-h3-audio-T8"
    sampling_path = package_root / "sampling.py"
    if not sampling_path.is_file():
        raise RuntimeError(
            "T8 dual-clock backend requires custom_nodes/minimax-h3-audio-T8. "
            "Install T8 or switch sampling_backend to native."
        )

    package_name = "_minimax_h3_t8_director_bridge"
    module_name = f"{package_name}.sampling"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    package = types.ModuleType(package_name)
    package.__path__ = [str(package_root)]
    sys.modules[package_name] = package
    spec = importlib.util.spec_from_file_location(module_name, sampling_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the T8 dual-clock sampler module.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _unpack_node_output(out):
    if hasattr(out, "args"):
        args = out.args
        if args:
            return args
    if isinstance(out, (tuple, list)):
        return out
    raise RuntimeError(f"Unexpected node output type: {type(out)!r}")


def sample_single_stage(
    *,
    model,
    positive,
    negative,
    latent,
    seed: int,
    cfg: float,
    steps: int,
    sampler_name: str,
    scheduler: str,
    shift_video: float = 12.0,
    shift_audio: float = 3.0,
    sampling_backend: str = "native",
    on_phase: PhaseCallback | None = None,
):
    def notify(phase: str, value: float) -> None:
        if on_phase:
            on_phase(phase, value)

    notify("sample", 0)

    if sampling_backend == "t8_dual_clock":
        from comfy_extras.nodes_custom_sampler import (
            BasicGuider,
            Noise_RandomNoise,
            SamplerCustomAdvanced,
        )

        if cfg != 1.0:
            log.warning("T8 dual-clock sampling uses BasicGuider; cfg=%s is ignored (H3 recommended value is 1.0).", cfg)

        t8_sampling = _load_t8_sampling_module()
        patched_model, custom_sampler, sigmas = t8_sampling.setup_dual_clock_sampling(
            model, latent, int(steps), float(shift_video), float(shift_audio)
        )
        guider = BasicGuider.execute(patched_model, positive).args[0]
        sampled = SamplerCustomAdvanced.execute(
            Noise_RandomNoise(int(seed)), guider, custom_sampler, sigmas, latent
        ).args[0]
        notify("sample", 1)
        return sampled

    if sampling_backend != "native":
        raise ValueError(f"Unknown MiniMax H3 sampling backend: {sampling_backend}")

    from comfy_extras.nodes_minimax_h3 import MiniMaxH3SigmaShift
    from nodes import KSampler

    shifted = MiniMaxH3SigmaShift.execute(model, float(shift_video), float(shift_audio))
    model_shifted = _unpack_node_output(shifted)[0]

    neg = negative if negative else []
    sampler = KSampler()
    samples, = sampler.sample(
        model_shifted,
        int(seed),
        int(steps),
        float(cfg),
        sampler_name,
        scheduler,
        positive,
        neg,
        latent,
        denoise=1.0,
    )
    notify("sample", 1)
    return samples
