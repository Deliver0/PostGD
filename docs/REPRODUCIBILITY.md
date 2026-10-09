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
