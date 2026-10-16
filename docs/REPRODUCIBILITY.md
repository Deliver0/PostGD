# Server-run reproducibility checklist

The original experiments were executed on a server. This checklist keeps the
public repository independent from server-specific paths while preserving the
experiment contract.

## Before stage 1

- [ ] Install `stage1/full_method/requirements.txt`.
- [ ] Download the [ReSDv1.4 fine-tuned UNet](https://zenodo.org/api/records/10947092/files/sd-retina-model.zip/content)
      and extract it so `models/sd-retina-model/checkpoint-60000/unet/` exists.
      The base `CompVis/stable-diffusion-v1-4` components are loaded from the
      Hugging Face cache (or downloaded with the explicit `--allow-download`
      flag).
- [ ] For the formal prior path, download the [LWNet DRIVE binary checkpoint](https://cdn.jsdelivr.net/gh/agaldran/lwnet@ce72ddabcf5fc9af14197ac0d43bd29000b17df2/experiments/wnet_drive/model_checkpoint.pth)
      to `engine/experiments/wnet_drive/model_checkpoint.pth`. The exact
      formal server run used this `wnet` checkpoint at 512px. Its pinned source
      is the [LWNet GitHub file](https://github.com/agaldran/lwnet/blob/ce72ddabcf5fc9af14197ac0d43bd29000b17df2/experiments/wnet_drive/model_checkpoint.pth).
      The historical GUI's `big_wnet` DRIVE/HRF pair is a separate path.
- [ ] Install the upstream `fundus_image_toolbox`; its fovea/optic-disc
      loader downloads [Zenodo weights](https://zenodo.org/records/11174642)
      when the configured checkpoint directory is absent, or explicitly use
      `--no-prior`.
- [ ] Fit the Lab profile using training reference images only.
- [ ] Store the generated data outside the repository.

## Stage 1 record

Record:

```text
command
git commit
background model identifier
background checkpoint identifier
color-profile path and hash
vessel/fovea checkpoint identifiers
seed
count
output size
spot profile
stage1_manifest.jsonl
```

## Before stage 2

- [ ] Verify the number of `*_synth.png` or `synth_*.png` images.
- [ ] Use the same image directory for the complete epoch.
- [ ] Keep teacher and student checkpoint identifiers.
- [ ] Write `config.json` before training.

## Stage 2 record

Keep:

```text
config.json
epoch_metrics.jsonl
last_checkpoint.pt
student_final.pth
student_to_teacher_projection.pth
```

## Stage 3 record

Keep the support/query manifest separate from the private image files. Record:

```text
backbone checkpoint hash
support sample identifiers
query sample identifiers
candidate list
CV seeds
selected candidate
threshold
per-sample predictions
final metrics
```

Do not select a candidate, epoch, or threshold using query/external labels.

### Classification head

```bash
python downstream/adapt_head.py \
  --support-npz /path/to/support_and_query.npz \
  --output ./runs/downstream_result.json
```

The result JSON keeps the candidate list, the selected entry, and the query metrics,
so the search space is auditable after the fact.

### Segmentation head

```bash
python downstream/segmentation/few_shot_segmentation.py \
  --manifest /path/to/dataset/manifest.json \
  --student-weights /path/to/student_final.pth \
  --backbone vit_base_patch16_dinov3 \
  --image-size 1024 \
  --output ./runs/seg_result
```

`summary.json` in the output directory already records most of the checklist above:

```text
backbone, image size, model kwargs
student SHA-256 (and the checkpoint dict key it was stripped from)
manifest SHA-256
epoch / threshold candidate lists
support-only leave-one-out CV scores for every candidate
selected epoch and threshold
per-task query metrics, written per image in test_metrics.jsonl
torch and python versions
```

Two protocol points matter when comparing runs:

- The support-only search must never see a query label. The released implementation
  runs exactly `len(support)` folds, each holding out one support image.
- The mask must live on the same grid the decoder produces, which is 16x the backbone
  token grid. `--preprocess-mode` (`stretch`, `center_crop` or `letterbox`) decides
  how a non-square source is mapped, and the same mode is used for images and masks.

Reproduce the unit tests with:

```bash
python -m pytest downstream/segmentation/tests -q
```

