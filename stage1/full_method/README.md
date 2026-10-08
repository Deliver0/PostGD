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
  [`wnet_drive/model_checkpoint.pth`](https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/wnet_drive/model_checkpoint.pth),
  placed at `engine/experiments/wnet_drive/model_checkpoint.pth`.
- Fovea/optic-disc prior: the upstream
  [Fundus Image Toolbox Zenodo weights](https://zenodo.org/records/11174642),
  which the toolbox can download when absent.

The clean public release does not include the historical `engine.models` and
`engine.utils` loader used by that server run. The checkpoint links are valid,
but the anatomy-prior path requires that compatible loader dependency as well.

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

The full 20,000-image server configuration is documented in the repository root
README. The output manifest records the seed, profile, requested spot upper
bound, and actual generated spot count for each image.
