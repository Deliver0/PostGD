# Complete stage-one pipeline

This directory implements background generation, Lab calibration,
anatomy-aware placement, spot rendering, FOV handling, and `224 x 224` export.

The default generator expects externally supplied model assets. Use
`--background-dir` to reuse existing backgrounds, and `--no-prior` only for a
deliberate geometry-only run.

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
