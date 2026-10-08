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
PostGD/
├── README.md                         Project overview and reproduction entry points
├── LICENSE                           MIT license for the released method code
├── CITATION.bib                      Citation information
├── configs/
│   └── hospital_lab_template.json    Public Lab-profile template
├── stage1/                            Synthetic image generation
│   ├── full_method/                  Complete background-to-image pipeline
│   │   ├── generate.py               Background, calibration, rendering, and export
│   │   ├── fit_lab_profile.py        Fit a profile from training images
│   │   ├── requirements.txt
│   │   └── src/
│   │       ├── color_calibration.py
│   │       ├── prior_guidance.py
│   │       └── spot_renderer.py
│   └── render_only/                  Renderer for an existing fundus background
│       ├── render_spots.py           Command-line entry point
│       ├── requirements.txt
│       ├── src/
│       │   ├── prior_guidance.py
│       │   └── spot_renderer.py
│       └── tests/
│           └── test_renderer.py
├── distillation/                     Teacher–student representation learning
│   ├── distill.py
│   └── requirements.txt
├── downstream/                       Few-shot downstream head adaptation
│   ├── adapt_head.py
│   └── README.md
└── docs/                             Reproducibility and implementation notes
    ├── IMPLEMENTATION.md
    ├── REPRODUCIBILITY.md
    └── THIRD_PARTY.md
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

No model checkpoint is bundled with or hosted by this repository. The links below
point to upstream sources; access, download, and use remain subject to each
source's terms.

| Component | Upstream source | Availability and use |
|---|---|---|
| Stable Diffusion v1.4 base components | [Hugging Face model](https://huggingface.co/CompVis/stable-diffusion-v1-4) | Use is subject to [CreativeML Open RAIL-M](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE). The generator downloads only when `--allow-download` is set. |
| Retina fine-tuned background UNet (DERETFound/ReSDv1.4) | [Zenodo record 10947092](https://zenodo.org/records/10947092), file [`sd-retina-model.zip`](https://zenodo.org/api/records/10947092/files/sd-retina-model.zip/content) | Directly downloadable. The tested file is `checkpoint-60000/unet/diffusion_pytorch_model.bin` (Zenodo MD5 `76e5aee5c54c79bdd0260aa344dc72fe` for the ZIP). The record declares CC-BY-4.0; Stable Diffusion Open RAIL-M restrictions also apply. |
| DINOv3 ViT-B/16 initialization | [Meta access page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), [official Hugging Face collection](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009), [official repository](https://github.com/facebookresearch/dinov3) | Gated: request access and accept Meta's [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). The repository does not mirror these weights. |
| PostGD distilled student | No public download | `student_final.pth` is a PostGD training output and is not currently published. Reproduce it with Stage 2 after obtaining permitted DINOv3 initialization weights. |
| Vessel prior used by the formal 20k run | [LWNet project](https://github.com/agaldran/lwnet), [`wnet_drive/model_checkpoint.pth`](https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/wnet_drive/model_checkpoint.pth) | Directly downloadable. This is the binary W-Net (`model_name=wnet`, 512px, DRIVE) used by the server's `stage1_20k` run; the SHA-256 of that upstream file is `91f0cada4b26ece63464b05be60f9b3a51f1bcd1f764081d07d3a35961118de1`. |
| Historical GUI vessel models | [DRIVE A/V checkpoint](https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/big_wnet_drive_av/model_checkpoint.pth), [HRF A/V checkpoint](https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/big_wnet_hrf_av_1024/model_checkpoint.pth) | Directly downloadable from the same upstream repository. These `big_wnet` models (512px/1024px) were used by the historical `xingai.py` GUI, not by the formal `stage1_20k` path. |
| Fovea/optic-disc localization | [Fundus Image Toolbox](https://github.com/berenslab/fundus_image_toolbox), [Zenodo weights](https://zenodo.org/records/11174642) | The Zenodo record is open access but does not declare a checkpoint license. Do not redistribute these weights or treat them as license-cleared without confirmation from the authors. |

The model names accepted by this code do not grant access to weights. In
particular, the DINOv3 teacher initialization is not the PostGD student
checkpoint, and no PostGD-trained student checkpoint is currently available
for direct download.

### Verified Stage 1 model layout

The server run recorded in `stage1_20k/_stage1_config.json` had
`blood_vessel_avoidance=true`. Its log reports `vessel=ok` and `fovea_od=ok`
for all four backgrounds. The corresponding local layout is:

```text
models/sd-retina-model/checkpoint-60000/unet/
  config.json
  diffusion_pytorch_model.bin
engine/experiments/wnet_drive/
  model_checkpoint.pth
engine/fit_models/2024-05-07 11_13.05/
  multi_efficientnet-b3_best.pt
```

The fovea/optic-disc file is downloaded by
`fundus_image_toolbox.load_fovea_od_model()` from [Zenodo record
11174642](https://zenodo.org/records/11174642) when the configured local
directory is absent. The public repository does not mirror these files.

For the two model assets used by the formal Stage 1 path, the upstream files
can be obtained directly (Linux/macOS example):

```bash
mkdir -p models engine/experiments/wnet_drive
curl -L --fail \
  https://zenodo.org/api/records/10947092/files/sd-retina-model.zip \
  -o /tmp/sd-retina-model.zip
unzip -q /tmp/sd-retina-model.zip -d models
curl -L --fail \
  https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/wnet_drive/model_checkpoint.pth \
  -o engine/experiments/wnet_drive/model_checkpoint.pth
```

The first archive should leave
`models/sd-retina-model/checkpoint-60000/unet/diffusion_pytorch_model.bin`.
The W-Net checkpoint is loaded by the historical `engine.models` implementation
used in the formal server run; that implementation is not mirrored in this
minimal public repository, so the download does not by itself add the missing
Python loader.

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

`--no-prior` disables vessel segmentation, fovea localization, and optic-disc
localization. It is a geometry-only rendering path; spots may overlap vessels
or other anatomy. It does not provide automatic anatomical detection.

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

Prior checkpoints and the clinical training data used to obtain them are not
included in this repository. The current adapter also depends on model code
that is not bundled here, so downloading a checkpoint alone does not make the
anatomy-aware path ready to run. See [`docs/THIRD_PARTY.md`](docs/THIRD_PARTY.md)
for compatibility and licensing details.

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

The following are not included in this repository:

- hospital or patient images;
- external validation images and labels;
- Stable Diffusion fine-tuning checkpoints;
- DINOv3 checkpoints;
- vessel/fovea localization checkpoints;
- generated image collections and experiment logs.

Links to upstream checkpoints are not a blanket license or legal guarantee.
Review the exact model, base-model, and dataset terms before use or
redistribution. In particular, the fovea/optic-disc checkpoint has no license
declared in its Zenodo record, and the PostGD distilled student is not publicly
hosted.

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
