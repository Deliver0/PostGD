# PostGD

**Generation–Distillation adaptation for post-photocoagulation fundus images.**

PostGD is a three-stage research pipeline for adapting a vision foundation model to
fundus images containing post-photocoagulation laser spots:

1. **Anatomy-aware synthetic image generation**
2. **Teacher–student representation distillation**
3. **Few-shot downstream head adaptation**

The repository contains the method implementation and reproducibility utilities.
Clinical images, derived hospital statistics, model checkpoints, and generated
datasets are intentionally kept outside Git.

> This is research code. It is not a medical device and must not be used for
> diagnosis or treatment decisions.

## Method at a glance

```text
background image or diffusion-generated background
                |
                v
       Lab-domain calibration
                |
                v
 anatomy-aware spot placement and rendering
                |
                v
          224 x 224 images
                |
                v
       DINOv3 teacher–student distillation
                |
                v
        frozen adapted backbone
                |
                v
        few-shot classification head
```

## Repository layout

```text
stage1/
  full_method/       Complete background -> calibration -> rendering pipeline
  render_only/       Standalone renderer for an existing fundus background
distillation/         DINOv3 teacher–student distillation
downstream/            Dataset-agnostic few-shot head adaptation
configs/               Public configuration templates
docs/                  Reproducibility and implementation notes
```

## Installation

Use separate environments when possible because Stable Diffusion and DINOv3
distillation have different GPU/runtime requirements.

The released scripts were developed against Python 3.10/3.11 and CUDA-enabled
PyTorch. Exact CUDA wheels should be selected from the official PyTorch index
for the target machine.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r stage1/render_only/requirements.txt
```

For the complete stage-one pipeline:

```bash
pip install -r stage1/full_method/requirements.txt
```

For distillation:

```bash
pip install -r distillation/requirements.txt
```

## Pretrained models

No checkpoint is bundled with this repository. The formal experiments used:

| Component | Expected input | How to provide it |
|---|---|---|
| Stable Diffusion base components | 512px background generation | local Hugging Face cache or `--base-model` |
| Fine-tuned background UNet | SD-compatible UNet | `--weight-dir` |
| DINOv3 teacher/student | `vit_base_patch16_dinov3` | `--teacher-weights` and `--student-weights` |
| Vessel prior | W-Net-compatible checkpoint | `--vessel-model` or `POSTGD_VESSEL_MODEL` |
| Fovea/optic-disc prior | local toolbox checkpoint directory | `--fovea-model-dir` or `POSTGD_FOVEA_MODEL_DIR` |

The checkpoints are separate from the code license and must be obtained under
their own terms.

## Stage 1: synthetic image generation

### 1A. Complete pipeline

The complete pipeline can generate backgrounds with Stable Diffusion, calibrate
the non-black FOV region in Lab space, place spots subject to anatomical
avoidance, and export `224 x 224` images plus a JSONL manifest.

```bash
python stage1/full_method/generate.py \
  --count 20000 \
  --background-count 4 \
  --spot-case mixed \
  --output-size 224 \
  --device cuda \
  --output-dir /path/to/runs/stage1_20k \
  --resume
```

For a local smoke test using existing backgrounds:

```bash
python stage1/full_method/generate.py \
  --count 4 \
  --background-dir /path/to/backgrounds \
  --skip-color-cal \
  --no-prior \
  --output-size 224 \
  --output-dir /path/to/runs/stage1_smoke
```

The default 20,000-image configuration uses two approved spot profiles:

```text
bright_no_dark:
  spot_size=2.0, gap=0.001, gain=0.12
  dark_center=0, dark_to_bright=0, bright_zone=0.20
  fade_zone=0.80, dark_depth=0

dark_center:
  spot_size=1.8, gap=0.02, gain=0.10
  ring_center=0.55, ring_width=0.05
  ring_inner_fade=0.25, ring_outer_fade=0.45
  ring_shoulder_power=1.0, dark_depth=0.04
```

For a new dataset, fit the color profile on training reference images only:

```bash
python stage1/full_method/fit_lab_profile.py \
  --reference-dir /path/to/training_reference_images \
  --output /path/to/hospital_lab_profile.json
```

Pass the resulting file with `--color-profile`.

`--num-spots` is an upper bound, not a target that must be reached. Anatomical
avoidance, FOV boundaries, and inter-spot spacing take priority.

### 1B. Standalone renderer

The standalone renderer accepts an existing RGB fundus background:

```bash
python stage1/render_only/render_spots.py \
  --input /path/to/background.png \
  --output-dir ./output \
  --spot-size 2 \
  --gap 0.001 \
  --num-spots 160 \
  --dark-center 0 \
  --dark-to-bright 0 \
  --bright-zone 0.2 \
  --fade-zone 0.8 \
  --dark-depth 0 \
  --gain 0.12
```

Outputs include:

```text
*_synth.png
*_mask.png
*_meta.json
```

The renderer supports a ring profile with independent inner and outer shoulders.
Use `--no-prior` for a geometry-only smoke test. To enable anatomy-aware
placement, provide external prior-model paths through the command line or:

```text
POSTGD_VESSEL_MODEL=/path/to/wnet_model_directory
POSTGD_FOVEA_MODEL_DIR=/path/to/fovea_od_model_directory
```

The prior adapter is deliberately external: checkpoints and the clinical
training data used to obtain them are not included in this repository.

## Stage 2: teacher–student distillation

The distillation script consumes the `224 x 224` synthetic images from stage 1.
It uses two augmented views of each image:

```text
L = MSE(projector(student(view_1)), teacher(view_1))
    + beta * (1 - cosine(student(view_1), student(view_2)))
```

The teacher is frozen. The student and an optional feature projector are updated
with AdamW. The formal configuration is 200 epochs, batch size 64, learning rate
`1e-4`, weight decay `0.05`, warmup of 500 steps, `beta=0.1`, and no validation
split.

```bash
python distillation/distill.py \
  --data-dir /path/to/runs/stage1_20k/distill_224 \
  --output-dir /path/to/runs/dinov3_distill_20k \
  --teacher-model vit_base_patch16_dinov3 \
  --student-model vit_base_patch16_dinov3 \
  --teacher-weights /path/to/weights/vit_base_patch16_dinov3 \
  --student-weights /path/to/weights/vit_base_patch16_dinov3 \
  --device cuda \
  --epochs 200 \
  --batch-size 64 \
  --expected-images 20000 \
  --validation-fraction 0 \
  --num-workers 8
```

The script writes:

```text
config.json
epoch_metrics.jsonl
last_checkpoint.pt
student_final.pth
student_to_teacher_projection.pth
```

Weights are loaded from a user-provided local path. No foundation-model
checkpoint is redistributed here.

## Stage 3: few-shot downstream adaptation

The downstream adapter is intentionally dataset-agnostic. It consumes frozen
backbone features rather than hard-coding hospital paths or labels.

Prepare an NPZ file with:

```text
support_features: [N, D] float32
support_labels:   [N]    int64
query_features:   [M, D] float32
query_labels:     [M]    int64
```

Then run:

```bash
python downstream/adapt_head.py \
  --support-npz /path/to/support_and_query.npz \
  --output /path/to/runs/downstream_result.json
```

The adapter selects among cosine prototypes, shrinkage prototypes, a learnable
prototype head, and regularized logistic regression using repeated stratified
2-fold cross-validation on the support set only. Query labels are read only for
the final report.

This mirrors the released experimental protocol:

```text
frozen PostGD backbone
  -> support-only head selection
  -> refit selected head on all support features
  -> independent query/external evaluation
```

## Data, weights, and privacy

The following are not included:

- hospital or patient images;
- external validation images and labels;
- Stable Diffusion fine-tuning checkpoints;
- DINOv3 checkpoints;
- vessel/fovea localization checkpoints;
- generated image collections and experiment logs.

Before running on clinical data, verify that you have the required data-use
permissions. Fit Lab color profiles on the training reference set only; do not
use holdout or external images to estimate calibration statistics.

## Reproducibility notes

- The generation seed controls both background and rendering randomness.
- Per-image metadata records the requested and actually generated spot counts.
- A failed vessel prior is handled conservatively by blocking placement; use
  `--no-prior` only for a deliberate geometry-only experiment.
- The public repository does not claim that a requested spot count will always
  be reached.
- The released downstream adapter does not tune hyperparameters on query labels.

See [`docs/IMPLEMENTATION.md`](docs/IMPLEMENTATION.md) for the implementation
contract and [`docs/REPRODUCIBILITY.md`](docs/REPRODUCIBILITY.md) for a
server-run checklist. Third-party components and their licensing boundaries are
listed in [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md).

## Citation

See [`CITATION.bib`](CITATION.bib). The BibTeX entry is intentionally minimal
until the associated manuscript has a stable public bibliographic record.

## License

The original method code in this repository is released under the MIT License.
Third-party packages, checkpoints, and prior-model implementations retain their
respective licenses and terms.
