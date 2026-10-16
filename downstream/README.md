# Few-shot downstream heads

Stage 3 turns the adapted backbone into a task model from a handful of labelled
images. Two heads are released; both are dataset-agnostic, both freeze the backbone,
and both select their hyper-parameters on **support labels only**.

| Head | Entry point | Output |
|------|-------------|--------|
| Classification | [`adapt_head.py`](adapt_head.py) | per-class logits / accuracy |
| Segmentation | [`segmentation/few_shot_segmentation.py`](segmentation/few_shot_segmentation.py) | binary mask / Dice, IoU, mIoU |

## Classification

`adapt_head.py` consumes frozen feature arrays and a support/query split:

```bash
python downstream/adapt_head.py \
  --support-npz /path/to/support_and_query.npz \
  --output ./runs/downstream_result.json
```

The NPZ keys are `support_features`, `support_labels`, `query_features`, and
`query_labels`. Query labels are withheld until the final metric calculation.

## Segmentation

`segmentation/` reads images and masks directly through a JSON manifest, caches
pyramid features from the frozen backbone, and fits a small convolutional decoder:

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
