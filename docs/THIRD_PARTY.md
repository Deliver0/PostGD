# Third-party components

PostGD code is released under the repository `LICENSE`. Dependencies and
external checkpoints are separate works.

## Python dependencies

The runtime dependencies are listed in:

- `stage1/full_method/requirements.txt`
- `stage1/render_only/requirements.txt`
- `distillation/requirements.txt`

Please follow each package's own license and citation requirements.

## External model assets

The following assets are expected to be supplied by the user and are not
redistributed:

- Stable Diffusion base components;
- the fine-tuned Stable Diffusion UNet used for the background generator;
- DINOv3 teacher/student checkpoints;
- vessel segmentation checkpoints;
- fovea/optic-disc localization checkpoints.

The anatomy-prior adapter calls the user's compatible prior implementation when
its paths are configured. The repository does not include clinical training
images, prior-model checkpoints, or historical private `engine/` code.

## Data

No patient-level or hospital image data is part of this repository. Users must
verify data-use permission, de-identification, and institutional policy before
running the pipeline on clinical images.

