# MiniMax H3 Director — 示例工作流

拖入 ComfyUI 画布即可使用。需已安装本插件，且 ComfyUI 主干含 MiniMax H3（v0.30.0+）。

| 文件 | 任务 | UNET | 说明 |
|------|------|------|------|
| `minimax_h3_director_t2v.json` | t2v | fl2va | 文生音视频，可直接 Queue |
| `minimax_h3_director_fl2v.json` | fl2v | fl2va | 首尾帧；「添加一组」后上传首帧和/或尾帧（可只传尾帧） |
| `minimax_h3_director_r2v.json` | r2v | **ref2va** | 参考改视频；素材组：图片1–9 / 音频1–3 / 视频1–3 |
| `minimax_h3_director_v2v.json` | v2v | **ref2va** | 源视频编辑；导演台上传视频并分段（同 Bernini v2v） |
| `minimax_h3_director_rv2v.json` | rv2v | **ref2va** | 参考改视频；源视频 + 图片1–9 |
| `minimax_h3_director_external_groups_i2v.json` | fl2v | fl2va | 外部 Group（Image to Video）→ Combine → Director.`i2v_groups`；时长/素材以接线为准 |
| `minimax_h3_director_external_groups_r2v.json` | r2v | **ref2va** | 外部 Group（Reference to Video）→ Combine → Director.`r2v_groups`；可用「选择运行」勾选组序 |
| `minimax_h3_director_二采_加速.json` | r2v | **ref2va** | 外接 **MiniMax H3 Director Refine** → Director.`refine`（SIGMAS + H3 latent）。`images` 为二采后成片，`images_pre_refine` 为一采对比片 |
| `minimax_h3_standalone_latent_refine.json` | t2v | fl2va | 独立一采 → Direct Latent 二采；同一份 positive conditioning 分支复用；输出一采/二采视频并保存二采 AV latent |
| `minimax_h3_latent_queue_refine.json` | t2v | fl2va | Queue Load 按批次读取 latent+prompt → 重建 Conditioning → Direct 二采；同时保存 refined 队列 |

## 模型路径（与官方模板一致）

| 用途 | 文件名 | 目录 |
|------|--------|------|
| UNET (t2v/i2v/fl2v) | `minimax_h3_fl2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| UNET (r2v / v2v / rv2v) | `minimax_h3_ref2va_pruned_int8_convrot.safetensors` | `models/diffusion_models/` |
| CLIP | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` | `models/text_encoders/` |
| Video VAE | `minimax_h3_video_vae_fp16.safetensors` | `models/vae/` |
| Audio VAE | `minimax_h3_audio_vae_fp32.safetensors` | `models/vae/` |

CLIP Loader 的 **type 必须选 `minimax`**。

## 默认采样参数

- 画布默认 **0.4MP 16:9（864×480）**，**5 秒 / 124** 帧 @ **24 fps**（17k+5 网格）
- **25** steps，`res_multistep` + `simple`，CFG **1.0**
- Sigma shift：video **12** / audio **3**

## 输出

导演台 → `CreateVideo` → `SaveVideo`（前缀 `video/MiniMaxH3_Director_*`），报告接 `PreviewAny`。

Refine 示例另把 `images_pre_refine` 接到第二路 `CreateVideo` / `SaveVideo`，方便和二采后成片对比。未接 Refine 时该口与 `images` 相同。

## Refine 二采

- 二采一律按 SIGMAS：把 `BasicScheduler` 或 `ManualSigmas` 接到 Refine 的 `sigmas` 口。`BasicScheduler` 请接和二采相同的 MODEL
- 不接 Refine 节点 = 原来的单次采样
- `mode=refine`：同分辨率精修；`mode=upscale`：放大到目标画布再二采；`mode=latent_upscale`：只放大 H3 latent
- 导演台分辨率是一采；Refine 画布（比例+百万像素 / 自定义）是放大目标
- `passes`：精修次数（默认 1）；`upscale` 只在第 1 次放大；`latent_upscale` 不二采
- 可选接 `refine_model`（二采 UNET）；不接则用导演台主模型
- `upscale` 默认 `h3_latent`：在 Refine 节点 `upscale_method` 下方下拉选 3D 权重（`mode=latent_upscale` 时同样出现）。权重放 `ComfyUI/models/latent_upscale_models/`。`lanczos` 可另接 `upscale_model`（RealESRGAN 等），不接则纯插值；也可改 `nvidia_rtx_vsr`
- fl2v 默认跳过二采；关掉 `skip_fl2v` 才会采首尾帧镜头

## 独立 Latent 二采工作流

`minimax_h3_standalone_latent_refine.json` 不使用导演台时间轴：

```text
MiniMaxH3DirectorConditioning
  ├─ positive → 官方 BasicGuider → 一采 SamplerCustomAdvanced
  └─ positive + 一采 LATENT → MiniMax H3 Latent Refine (Direct)
                                      ├─ VAEDecode / VAEDecodeAudio → 二采视频
                                      └─ MiniMax H3 Latent Save → output/minimax_h3_latents/
```

同一张工作流内不需要复制提示词；拆到另一张工作流时，需要重新建立 positive conditioning，r2v 还要重新接参考素材。

### Latent + Prompt 队列

新增 **MiniMax H3 Latent Queue Save / Load**：

```text
一采 H3 AV LATENT + Conditioning.prompt
        └→ Queue Save → output/minimax_h3_latents/queues/<name>/0001__shot__latent.mmxlatent.zip

Queue Load (next / index, batch_size)
        ├→ prompt → MiniMaxH3DirectorConditioning.prompt
        └→ latent → MiniMax H3 Latent Refine (Direct)
```

每个队列包内嵌一个 `queue_record`，包括 prompt、编号和文件标识；读取器始终成对输出，不按两个独立文件列表拼接。`next` 模式每次 Queue Prompt 顺序消费，`reset_token` 加 1 可重置游标；默认一次一条，适合无卡启动和显存有限的 H3 二采。

导演台也可以直接填写可选输入 `latent_queue_name`（如 `h3_first_pass`）自动按片段导出一采队列；`latent_queue_stage=final` 则导出导演台已经完成的终稿 latent。未填写队列名时，导演台行为不变。
