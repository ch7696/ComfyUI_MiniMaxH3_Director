# Deployment Notes

This repository contains code only. Do not add model weights, generated media, credentials, or a local Python virtual environment.

## Pinned Components

- Upstream Director base: `1b38725312af88bfb6686e981b74b98b130e5794`
- T8: `a9e86b7810693055dc19861ff954911fc7a6098a`
- ComfyUI: v0.30.0 or newer with official MiniMax H3 nodes

## Expected Layout

```text
ComfyUI/
  custom_nodes/
    ComfyUI_MiniMaxH3_Director/
    minimax-h3-audio-T8/
  models/                 # mount from persistent storage
  input/                  # per-job staging
  output/                 # upload completed jobs to object storage
```

Keep model files on a persistent SSD or network volume and configure them through `extra_model_paths.yaml`. Keep secrets in environment variables or the cloud provider's secret manager.

Do not expose ComfyUI port 8188 directly to the public internet. Put an authenticated controller or reverse proxy in front of the ComfyUI API.
