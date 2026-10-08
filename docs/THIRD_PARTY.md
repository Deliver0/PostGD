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

No model checkpoint is bundled with or hosted by PostGD. The sources below are
upstream downloads, not PostGD releases. Download access does not by itself
establish permission to use or redistribute a checkpoint.

| Asset | Upstream source | License and availability | Compatibility notes |
|---|---|---|---|
| Stable Diffusion v1.4 base model | [CompVis Hugging Face model](https://huggingface.co/CompVis/stable-diffusion-v1-4) | CreativeML Open RAIL-M; review the [license](https://github.com/CompVis/stable-diffusion/blob/main/LICENSE) and its use restrictions. | The generator's default base model is `CompVis/stable-diffusion-v1-4`. Network download is opt-in with `--allow-download`. |
| Retina fine-tuned SD UNet | [Zenodo record 10947092](https://zenodo.org/records/10947092), [`sd-retina-model.zip`](https://zenodo.org/api/records/10947092/files/sd-retina-model.zip/content), DOI [10.5281/zenodo.10947092](https://doi.org/10.5281/zenodo.10947092) | The Zenodo record declares CC-BY-4.0. The checkpoint is based on Stable Diffusion, whose Open RAIL-M restrictions also apply. Keep attribution and pass on applicable use restrictions. | Extract locally. The generator looks for `checkpoint-60000/unet/` under the path passed as `--weight-dir`. |
| DINOv3 ViT-B/16 initialization | [Meta download access page](https://ai.meta.com/resources/models-and-libraries/dinov3-downloads/), [official Hugging Face collection](https://huggingface.co/collections/facebook/dinov3-68924841bd6b561778e31009), [official repository](https://github.com/facebookresearch/dinov3) | Gated access. A user must request access and accept the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md). Do not mirror the gated files here. | Use an authorized ViT-B/16 checkpoint for teacher and student initialization. The repository's `student_final.pth` output is a separate PostGD-trained artifact and is not publicly hosted. |
| W-Net vessel segmentation | [LWNet project and checkpoints](https://github.com/agaldran/lwnet), [DRIVE W-Net checkpoint](https://raw.githubusercontent.com/agaldran/lwnet/master/experiments/wnet_drive/model_checkpoint.pth) | The upstream code repository declares MIT. Checkpoint-specific terms and training-dataset rights were not independently established; retain upstream notices and verify before redistribution. | Not verified as drop-in compatible. The current adapter imports `engine.models` and `engine.utils` code that is not included in this repository, and expects a binary `wnet` model. A checkpoint alone is insufficient. |
| Fovea/optic-disc localization | [Fundus Image Toolbox](https://github.com/berenslab/fundus_image_toolbox), [Zenodo record 11174642](https://zenodo.org/records/11174642), [weights.tar.gz](https://zenodo.org/api/records/11174642/files/weights.tar.gz/content), DOI [10.5281/zenodo.11174642](https://doi.org/10.5281/zenodo.11174642) | The software repository declares MIT, but the Zenodo record does not declare a license for the checkpoint. Obtain clarification from the authors before redistribution or treating it as license-cleared. | Requires the upstream toolbox and a locally extracted checkpoint directory; the current prior adapter is not self-contained. |

The repository does not include clinical training images, prior-model
checkpoints, or historical private `engine/` code. The prior adapter's
`--vessel-model` and `--fovea-model-dir` options only identify local paths; they
do not download, validate, or make arbitrary upstream weights compatible.

## Redistribution guidance

- Do not commit model checkpoints to this repository unless their exact
  redistribution terms and any inherited base-model restrictions have been
  reviewed.
- Follow the DINOv3 access agreement and license for authorized downloads.
- Preserve attribution and the use restrictions that apply to Stable Diffusion
  derivatives.
- The fovea/optic-disc checkpoint has no license declared in the Zenodo record;
  ask its authors before redistribution.
- Do not describe the anatomy-aware path as ready to use with only these
  downloads: its model implementation dependencies are not bundled, and the
  linked checkpoints have not been validated against this adapter.

## Data

No patient-level or hospital image data is part of this repository. Users must
verify data-use permission, de-identification, and institutional policy before
running the pipeline on clinical images.
