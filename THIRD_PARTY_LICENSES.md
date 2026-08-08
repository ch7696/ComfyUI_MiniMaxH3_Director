# Third-Party Software

## AIMixer ComfyUI MiniMax H3 Director

- Source: https://github.com/AIMixer/ComfyUI_MiniMaxH3_Director
- License: Apache-2.0
- Local base commit: `1b38725312af88bfb6686e981b74b98b130e5794`
- Changes: optional T8 backend selection, T8 sampler loading, and execution routing

The upstream Apache-2.0 license and attribution are retained in [LICENSE-APACHE-2.0](LICENSE-APACHE-2.0) and [NOTICE](NOTICE).

## MiniMax H3 Audio T8

- Source: https://github.com/T8mars/comfyui-minimax-h3-audio-T8
- License: GPL-3.0-or-later
- Tested commit: `a9e86b7810693055dc19861ff954911fc7a6098a`

T8 is not copied into this repository. The `t8_dual_clock` backend loads its `sampling.py` module from the separately installed sibling directory `custom_nodes/minimax-h3-audio-T8`. This fork is distributed under GPL-3.0-or-later to preserve compatibility with that runtime integration.

## Model Weights

No model weights are distributed by this repository. Users must obtain MiniMax H3, Qwen3-VL, VAE, and LoRA weights from their original sources and comply with the applicable model licenses.
