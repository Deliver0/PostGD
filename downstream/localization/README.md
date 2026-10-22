# Few-shot landmark localisation head

`few_shot_localization.py` fits a light heat-map head on top of a **frozen** backbone
and regresses normalised `(x, y)` coordinates for every landmark, using only a handful
of labelled images. It is the localisation counterpart of the segmentation decoder in
[`../segmentation/`](../segmentation/) and shares the same contract: the query split is
never used for model selection.

```bash
python downstream/localization/few_shot_localization.py \
  --manifest /path/to/dataset/landmarks.json \
  --student-weights /path/to/student_final.pth \
  --backbone vit_base_patch16_dinov3 \
  --image-size 1024 \
  --output /path/to/runs/localization_result
```

`--image-size` defaults to `224`, which is the resolution used by the released
classification head; set it to `1024` to match the segmentation head and Table III.

## What runs

1. Every image is pushed through the frozen backbone once and its token sequence is
   cached to memory (`build_feature_sets`). Photometric jitter and a horizontal flip
   create `--aug-repeats` extra views; a flipped view carries a mirrored target, so an
   augmentation can never move a landmark.
2. For each task in the manifest, the epoch count is chosen by **support-only
   leave-one-out cross-validation** (`select_epoch`). Query annotations are not read at
   this point.
3. The head is retrained on the full support set with the selected epoch count and
   evaluated once on the query split.
4. `summary.json` records the selected settings, the query metrics and a full provenance
   block (backbone name, student SHA-256, manifest SHA-256, every hyper-parameter,
   `torch` and `python` versions).

`test_metrics.jsonl` holds one line per query image so per-sample predictions can be
audited after the fact.

## Manifest format

A single JSON file describes the dataset. No NPZ export, no fixed directory convention:

```json
{
  "coord_space": "normalized",
  "samples": [
    {"image_id": "02", "image": "images/02.jpg",
     "landmarks": {"fovea": [0.412, 0.523], "optic_disc": [0.771, 0.468]}}
  ],
  "tasks": {
    "fovea": {"support": ["02", "05", "07", "11", "13"], "test": ["17", "19"]},
    "both": {"support": ["02", "05", "07", "11", "13"], "test": ["17", "19"],
             "landmarks": ["fovea", "optic_disc"]}
  }
}
```

Paths are absolute, or relative to the manifest's own directory (or to `--data-root`
when given). `coord_space` says whether annotations are fractions of the source image
(`normalized`, the default) or absolute pixels (`pixel`). A task without a `landmarks`
list regresses the annotation of the same name; a task with one regresses several
landmarks jointly with a single head. `--tasks` runs a subset.

## Head

`LocalizationHead` folds the token sequence back into a `H x W` grid, projects every
token to `--hidden-dim` with a 1x1 convolution, mixes the grid with a 3x3 convolution
and emits one heat map per landmark. `soft_argmax_2d` takes the spatial expectation of
each heat map, so the output is a differentiable `(x, y)` in `[0, 1]` — grid cell `i`
of `W` sits at `i / (W - 1)`, which lets the head reach both borders of the square.

Training minimises a `--loss-mode smooth_l1` (`--beta` sets the transition point) or
`mse` loss on normalised coordinates, optionally plus `--l1-weight` times the L1 norm of
the head weights. Gradients are clipped to a norm of 5.0, matching the segmentation
decoder.

## Metric

Reported distance is the Euclidean distance between the predicted and the annotated
point in pixels **at the input resolution**: `||p - q||_2 * image_size`. This is the
quantity tabulated in Table III, so results at `--image-size 1024` are directly
comparable with the paper. Each row carries the per-landmark distance
(`<name>_ed_px`), plus `mean_ed_px`, `median_ed_px`, `max_ed_px` and
`mean_ed_normalized`; `summary.json` averages those over the query set.

## Preprocessing

`--preprocess-mode` selects how a non-square source is mapped onto the network square:
`stretch` (direct resize, the default), `center_crop` or `letterbox`. Landmarks go
through the same geometry as the pixels, so a coordinate that falls outside a crop is
clipped to the square rather than silently mis-regressed. Use the mode that the
backbone was distilled with.

## Loading a student checkpoint

`--student-weights` points at the `.pth` produced by Stage 2. Both a raw `state_dict`
and a checkpoint dictionary are accepted. When it is omitted the backbone falls back to
the architecture named by `--backbone` with random weights, which is only useful for a
smoke test.

Some checkpoints were trained at a resolution other than 224. Forward the working
resolution to `timm` so the positional embedding is resized consistently:

```bash
--model-kwargs '{"img_size": 1024}'
```

## Tests

```bash
python -m pytest downstream/localization/tests -q
```

The suite runs on CPU in seconds with a stub backbone, and covers the geometry of the
three preprocessing modes, the head's forward shape, loss back-propagation, the
hand-computable Euclidean distance, the five-shot one-epoch training path and the
leak-free property of the support-only search.
