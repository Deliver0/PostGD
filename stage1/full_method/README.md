# Complete stage-one pipeline

This directory implements background generation, Lab calibration,
anatomy-aware placement, spot rendering, FOV handling, and `224 x 224` export.

The default generator expects externally supplied model assets. Use
`--background-dir` to reuse existing backgrounds, and `--no-prior` only for a
deliberate geometry-only run.

The formal server run used the following directly obtainable assets:

- Backgrounds: DERETFound/ReSDv1.4 fine-tuned UNet from
  [Zenodo `sd-retina-model.zip`](https://zenodo.org/api/records/10947092/files/sd-retina-model.zip/content),
  extracted as `models/sd-retina-model/checkpoint-60000/unet/`, together with
  the `CompVis/stable-diffusion-v1-4` base components.
- Vessel prior: the binary DRIVE W-Net checkpoint
  [`wnet_drive/model_checkpoint.pth` source](https://github.com/agaldran/lwnet/blob/ce72ddabcf5fc9af14197ac0d43bd29000b17df2/experiments/wnet_drive/model_checkpoint.pth),
  with a [jsDelivr direct download](https://cdn.jsdelivr.net/gh/agaldran/lwnet@ce72ddabcf5fc9af14197ac0d43bd29000b17df2/experiments/wnet_drive/model_checkpoint.pth),
  placed at `models/wnet_drive/model_checkpoint.pth` and supplied with `--vessel-model`.
- Fovea/optic-disc prior: the upstream
  [Fundus Image Toolbox Zenodo weights](https://zenodo.org/records/11174642),
  downloaded in advance and supplied as an existing `--fovea-model-dir`.

The release includes a self-contained W-Net loader in `src/wnet_vessel.py`.
The vessel checkpoint and installed Fundus Image Toolbox remain external
requirements. Missing anatomical priors stop generation; use `--no-prior`
only for an explicit geometry-only run.

Fit a profile from training reference images before calibration:

```bash
python fit_lab_profile.py \
  --reference-dir /path/to/training_reference_images \
  --output /path/to/hospital_lab_profile.json
```

Then pass it to `generate.py` with `--color-profile`.

```bash
python generate.py \
  --count 1 \
  --background-dir /path/to/backgrounds \
  --skip-color-cal \
  --no-prior \
  --output-dir ./smoke
```

The full 200,000-image server configuration is documented in the repository root
README. The output manifest records the seed, profile, requested spot upper
bound, and actual generated spot count for each image.

## Treatment mode

`--treatment prp` is the default: optic-disc-centred rings outside the
fovea-centred posterior region. `--treatment grid` uses fovea-centred rings
inside that region. Both avoid the vessel and optic-disc masks and require
centres to lie inside the retinal field of view.

```bash
python stage1/full_method/generate.py \
  --count 10 --treatment grid \
  --background-dir /path/to/backgrounds --skip-color-cal \
  --vessel-model ./models/wnet_drive \
  --fovea-model-dir /path/to/fovea_od/checkpoint-directory \
  --output-dir ./runs/grid
```

Run this example from the repository root. GRID requires working anatomical
priors; `--no-prior` is available for PRP geometry-only tests. Appearance presets,
ring spacing and spot-count limits are shared by both modes. The manifest and
run configuration record `treatment`; resume requires the same treatment.

Exported image/laser-mask pairs share the same FOV crop and circular masking.
Images use Lanczos resizing; labels use nearest-neighbor resizing. Images go
into `distill_<size>/`, masks into `laser_masks_<size>/`, and per-image
geometry/asset metadata into `metadata/`. The manifest records all three paths;
only `distill_<size>/` should be passed to distillation.
