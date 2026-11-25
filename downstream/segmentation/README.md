# Few-shot segmentation head

`few_shot_segmentation.py` fits a light segmentation decoder using a handful of
labelled images. OD/OC tasks also adapt the last one/two backbone blocks. It is the segmentation counterpart
of the classification adapter in [`../adapt_head.py`](../adapt_head.py) and shares the
same contract: the query split is never used for model selection.

```bash
python downstream/segmentation/few_shot_segmentation.py \
  --manifest /path/to/dataset/manifest.json \
  --student-weights /path/to/student_final.pth \
  --backbone vit_base_patch16_dinov3 \
  --image-size 1024 \
  --output /path/to/runs/seg_result
```

## What runs

1. OD/OC tasks compute features during training so gradients reach their selected
   backbone blocks. Other tasks cache the frozen four-level features (`build_feature_sets`).
2. For each task in the manifest, an epoch count and a binarisation threshold are
   chosen by **support-only leave-one-out cross-validation** (`select_epoch_and_threshold`).
   Query labels are not read at this point.
3. The decoder and selected backbone blocks are trained on the full support set
   with the selected epoch count, then evaluated once on the query split. Each
   validation fold and final fit starts from the same original student weights.
4. `summary.json` records the selected settings, the query metrics and a full
   provenance block (backbone name, student SHA-256, manifest SHA-256, every
   hyper-parameter, `torch` and `python` versions).

`test_metrics.jsonl` holds one line per query image so per-sample predictions can be
audited after the fact.

## Backbone adaptation

| Manifest task name | Trainable backbone blocks | Default backbone LR |
| --- | --- | --- |
| `optic_disc`, `OD`, `disc` | Last 1 | `1e-5` |
| `optic_cup`, `OC`, `cup` | Last 2 | `1e-5` |
| Other task names | None | Not applicable |

`optic_disc_segmentation` and `optic_cup_segmentation` are also recognized.
Task names are case-insensitive; spaces and hyphens normalize to underscores.
Use `--finetune-blocks N` to override the block count for all tasks in that run,
or `--finetune-blocks 0` to select the frozen-feature path. Decoder LR is set by
`--lr` (default `3e-4`), backbone LR by `--backbone-lr` (default `1e-5`).

```bash
python downstream/segmentation/few_shot_segmentation.py \
  --manifest /path/to/dataset/manifest.json \
  --student-weights /path/to/student_final.pth \
  --tasks optic_disc optic_cup \
  --lr 3e-4 --backbone-lr 1e-5 \
  --image-size 1024 --model-kwargs '{"img_size": 1024}' \
  --output /path/to/runs/od_oc
```

Adapted tasks save `<task>/task_model.pth` with the full task-specific backbone,
decoder, selected threshold and preprocessing settings. These task backbones
are independent; adapting one task does not change the shared starting checkpoint.
To reload in a script that can import the segmentation modules:

```python
import timm
import torch
from partial_finetune import PartialSegmentationModel

state = torch.load("optic_disc/task_model.pth", map_location="cpu", weights_only=True)
backbone = timm.create_model(state["backbone_name"], pretrained=False,
                             num_classes=0, **state["model_kwargs"])
model = PartialSegmentationModel(backbone, state["finetune_blocks"])
model.backbone.load_state_dict(state["backbone"])
model.decoder.load_state_dict(state["decoder"])
model.eval()
```

Use the saved `image_size`, `preprocess_mode`, `selected_threshold` and `tta`
settings when evaluating the reloaded model.

## Manifest format

A single JSON file describes the dataset. No NPZ export, no fixed directory
convention:

```json
{
  "samples": [
    {"image_id": "02", "image": "images/02.jpg",
     "masks": {"laser_spots": "masks/laser/02.png"}}
  ],
  "tasks": {
    "laser_spots": {"support": ["02"], "test": ["05", "07"]}
  }
}
```

Paths are absolute, or relative to the manifest's own directory. Several tasks can
share one image through the `masks` mapping, and `--tasks` runs a subset of them.

## Decoder

`SegDecoder` projects each pyramid level to a common width with a 1x1 convolution
(64 / 128 / 256 / 512 channels), fuses them coarse-to-fine, upsamples four times by
a factor of two and emits two-class logits. The output therefore sits on a
grid 16x finer than the backbone tokens, so masks must be resampled to match —
`load_mask_tensor` does that, using the same `--preprocess-mode` (`stretch`,
`center_crop` or `letterbox`) that maps non-square sources onto the square input.

Training minimises a weighted cross-entropy plus a Dice term
(`--loss-mode foreground` for single-class masks, `multiclass` when a mask carries
several class indices). Pass `--dice-weight` to rebalance, and `--tta` to enable
horizontal-flip test-time augmentation.

## Leak-free evaluation

`select_epoch_and_threshold` runs exactly `len(support)` folds, always leaving one
support image out and never touching a query label. The candidate list is fixed
before the run and written into `summary.json`, so the search space itself is part
of the record. Reported metrics are Dice, foreground IoU, background IoU, binary
mIoU and the predicted positive fraction.

## Loading a student checkpoint

`--student-weights` points at the `.pth` produced by Stage 2. The loader accepts both
a raw `state_dict` or a dictionary with a `model` entry. Supply weights for the
architecture named by `--backbone`. Omitting weights creates a random
initialization for software checks, not a pretrained PostGD model.

Some checkpoints were trained at a resolution other than 224. Forward the working
resolution to `timm` so the positional embedding is resized consistently:

```bash
--model-kwargs '{"img_size": 1024}'
```

## Tests

```bash
python -m pytest downstream/segmentation/tests -q
```

The suite runs on CPU in seconds with a stub backbone, and covers preprocessing,
decoder geometry, loss back-propagation, metric endpoints and the leak-free property
of the support-only search.
