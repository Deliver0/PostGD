# Few-shot downstream heads

Stage 3 turns the adapted backbone into a task model from a handful of labelled
images. Three heads are released; all are dataset-agnostic and select their
hyper-parameters on **support labels only**. Classification and localization use
frozen representations; OD/OC segmentation also adapts the final one/two blocks.

| Head | Entry point | Output |
|------|-------------|--------|
| Classification | [`adapt_head.py`](adapt_head.py) | per-class logits / accuracy |
| Segmentation | [`segmentation/few_shot_segmentation.py`](segmentation/few_shot_segmentation.py) | binary mask / Dice, IoU, mIoU |
| Localisation | [`localization/few_shot_localization.py`](localization/few_shot_localization.py) | normalised `(x, y)` / Euclidean distance in px |

## Classification

`adapt_head.py` consumes frozen feature arrays and a support/query split:

First extract features from the student checkpoint:

```bash
python downstream/extract_features.py \
  --manifest /path/to/classification.json \
  --student-weights /path/to/student_final.pth \
  --output ./runs/classification/features.npz
```

The JSON manifest uses image IDs and binary labels:

```json
{
  "samples": [
    {"image_id": "a", "image": "images/a.png", "label": 0},
    {"image_id": "b", "image": "images/b.png", "label": 1},
    {"image_id": "c", "image": "images/c.png", "label": 0},
    {"image_id": "d", "image": "images/d.png", "label": 1},
    {"image_id": "e", "image": "images/e.png", "label": 0},
    {"image_id": "f", "image": "images/f.png", "label": 1}
  ],
  "support": ["a", "b", "c", "d"],
  "query": ["e", "f"]
}
```

Image paths are relative to the manifest directory or absolute. For the paper's
ten-shot classification setting, provide five support images per class; the
example above only illustrates the format. Support and query IDs must be
disjoint. Extraction leaves the student frozen. A `.provenance.json` sidecar
records the student hash and preprocessing settings. Then fit the head:

```bash
python downstream/adapt_head.py \
  --support-npz /path/to/support_and_query.npz \
  --output ./runs/downstream_result.json
```

The NPZ keys are `support_features`, `support_labels`, `query_features`, and
`query_labels`. Query labels are withheld until the final metric calculation.

The result references a `.head.json` with fitted prototype or logistic-regression
parameters. To reuse it on features from the same student and preprocessing:

```python
import json
from adapt_head import predict_head

head = json.load(open("downstream_result.head.json", encoding="utf-8"))
probabilities = predict_head(head, query_features)
predictions = probabilities >= head["threshold"]
```

## Segmentation

`segmentation/` reads images and masks through a JSON manifest and fits a small
convolutional decoder. The `optic_disc` / `OD` tasks adapt the last transformer
block; `optic_cup` / `OC` adapt the last two. Decoder and backbone learning rates
default to `3e-4` and `1e-5`. Other tasks use cached frozen features. Pass
`--finetune-blocks 0` to use the frozen path explicitly.

```bash
python downstream/segmentation/few_shot_segmentation.py \
  --manifest /path/to/dataset/manifest.json \
  --student-weights /path/to/student_final.pth \
  --backbone vit_base_patch16_dinov3 \
  --image-size 1024 \
  --output /path/to/runs/seg_result
```

The epoch count and binarisation threshold come from support-only leave-one-out
cross-validation; the query split is scored once, after the decoder is locked. Every
run writes `summary.json` with the selected settings, the query metrics and a
provenance block (backbone, student SHA-256, manifest SHA-256, all hyper-parameters,
`torch` and `python` versions), plus `test_metrics.jsonl` with one line per query
image.

Install its dependencies with
[`segmentation/requirements.txt`](segmentation/requirements.txt) and see
[`segmentation/README.md`](segmentation/README.md) for the manifest format, the
decoder architecture and the evaluation protocol.

## Localisation

`localization/` reads a JSON manifest of image paths and `(x, y)` landmark
annotations, caches the frozen token sequence, and fits a light heat-map head that
regresses normalised coordinates:

```bash
python downstream/localization/few_shot_localization.py \
  --manifest /path/to/dataset/landmarks.json \
  --student-weights /path/to/student_final.pth \
  --backbone vit_base_patch16_dinov3 \
  --image-size 1024 \
  --output /path/to/runs/localization_result
```

The epoch count comes from support-only leave-one-out cross-validation; the query
split is scored once, after the head is locked. Reported distance is the Euclidean
distance in pixels at `--image-size`, matching the paper's Table III. Each run writes
`summary.json` with the selected settings, the query metrics and the same provenance
block as the segmentation head, plus `test_metrics.jsonl` per query image.

Install its dependencies with
[`localization/requirements.txt`](localization/requirements.txt) and see
[`localization/README.md`](localization/README.md) for the manifest format, the head
architecture and the metric definition.
