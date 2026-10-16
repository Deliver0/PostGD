# Few-shot segmentation head

`few_shot_segmentation.py` fits a light segmentation decoder on top of a **frozen**
backbone, using only a handful of labelled images. It is the segmentation counterpart
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

1. Every image is pushed through the frozen backbone once, and the four pyramid
   levels of tokens are cached to memory (`build_feature_sets`).
2. For each task in the manifest, an epoch count and a binarisation threshold are
   chosen by **support-only leave-one-out cross-validation** (`select_epoch_and_threshold`).
   Query labels are not read at this point.
3. The decoder is retrained on the full support set with the selected epoch count,
   and evaluated once on the query split.
4. `summary.json` records the selected settings, the query metrics and a full
   provenance block (backbone name, student SHA-256, manifest SHA-256, every
   hyper-parameter, `torch` and `python` versions).

`test_metrics.jsonl` holds one line per query image so per-sample predictions can be
audited after the fact.

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
a factor of two and emits a single-channel logit map. The output therefore sits on a
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
a raw `state_dict` and a checkpoint dictionary and strips the projection head, so the
same file can be reused. When it is omitted the backbone falls back to the
pretrained DINOv3 initialisation named by `--backbone`.

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
