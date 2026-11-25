# Implementation and training notes

This document describes the released training framework, supported options and
external assets. Hardware and data paths are configured by the user.

## Covered in the release

- Synthetic background generation, W-Net vessel priors and Fundus Image Toolbox
  localization, followed by PRP or macular GRID placement and rendering.
- A frozen DINOv3 ViT-L/16 teacher and pretrained ViT-B/16 student with a
  768-to-1024 linear projector and feature-MSE distillation.
- Consistency regularization on the unprojected student embeddings.
- Four-block, top-down segmentation decoding with bilinear upsampling and
  cross-entropy plus Dice loss. The decoder learning-rate default is `3e-4`.
- OD/OC segmentation adapts the final one/two transformer blocks at `1e-5`,
  recomputing features during training and saving independent task checkpoints.
- Image-to-feature export, support-only classification and saved classification
  heads, plus landmark localization from images.

Anatomical generation now stops if fovea/optic-disc localization fails, just as
it already stops when the vessel prior fails. `--no-prior` remains an explicit
geometry-only mode.

## Configuration and assets

| Area | Configuration | Usage notes |
| --- | --- | --- |
| Treatment mode | PRP outside the posterior region; GRID inside it | Select `--treatment prp` (default) or `--treatment grid`. GRID uses the foveal centre and requires anatomical priors. Appearance presets are independent of treatment mode. |
| Training environment | The reported experiments used eight GPUs and a global batch size of 640 | The released single-process framework defaults to batch size 64. Users configure batch size and device for their hardware; a fixed eight-GPU launcher is outside the release scope. |
| OD/OC segmentation adaptation | Last one/two blocks at `1e-5`; decoder at `3e-4` | Enabled by task name; use `--finetune-blocks` to override. All folds and final training start from the same student weights. |
| Student download | Pretrained ViT-B/16 backbone | Available on [Hugging Face](https://huggingface.co/Deliver0/postgd-dinov3-student) as PyTorch and Safetensors weights. The model card documents the artifact's provenance and associated configuration. |
| Classification extraction | Frozen shared student encoder followed by a classification head | Run `downstream/extract_features.py` to export support/query features, then `adapt_head.py`. The fitted head and feature provenance are saved. |
| EMA checkpoint | An EMA student is maintained by the runner | Both raw and EMA state dicts are written. The checkpoint used for the reported results still requires confirmation. |
| Model compatibility | Released DINOv3 weights and four intermediate feature maps | The published student passes strict CPU loading and an embedding forward check. Four-map extraction and partial-finetuning gradients were checked with a randomly initialized DINOv3 backbone. CUDA memory requirements and full downstream performance require validation in a configured training environment. |

The release does not include clinical images, patient annotations or published
per-image predictions. Support-only code paths do not verify the composition of
the private evaluation splits.
