# Model weights and local setup

This guide explains where each external checkpoint goes in a fresh clone. Model
files stay outside Git; the repository `.gitignore` already excludes common
checkpoint extensions. The commands below assume Linux/macOS and a shell opened
at the repository root. Replace `$POSTGD_ROOT` with the absolute clone path.

## 1. Create the local layout

```bash
cd /path/to/PostGD
export POSTGD_ROOT="$PWD"
export POSTGD_WEIGHTS="$POSTGD_ROOT/models"
mkdir -p "$POSTGD_WEIGHTS"
```

Use this layout. Directory names are part of the command examples below.

```text
models/
├── stable-diffusion-v1-4/
├── sd-retina-model/
│   └── checkpoint-60000/unet/
│       ├── config.json
│       └── diffusion_pytorch_model.bin
├── wnet_drive/
│   └── model_checkpoint.pth
├── fovea_od/
│   └── 2024-05-07 11_13.05/
│       └── multi_efficientnet-b3_best.pt
├── dinov3/
│   └── vit_base_patch16_dinov3.pth
└── student/
    └── student_final.pth
```

`stage1/full_method/generate.py` already defaults the retina UNet to
`models/sd-retina-model`. The other paths are supplied explicitly so that a
different clone or a shared storage volume can be used without editing code.

## 2. Retina background generator

The generator uses the DERETFound/ReSDv1.4 fine-tuned UNet from Zenodo and the
base tokenizer, text encoder, VAE, scheduler, and fallback UNet components from
Stable Diffusion v1.4.

Download and extract the fine-tuned UNet:

```bash
curl -L --fail \
  https://zenodo.org/api/records/10947092/files/sd-retina-model.zip/content \
  -o /tmp/sd-retina-model.zip
unzip -q /tmp/sd-retina-model.zip -d "$POSTGD_WEIGHTS"
test -f "$POSTGD_WEIGHTS/sd-retina-model/checkpoint-60000/unet/config.json"
test -f "$POSTGD_WEIGHTS/sd-retina-model/checkpoint-60000/unet/diffusion_pytorch_model.bin"
```

The tested archive has MD5 `76e5aee5c54c79bdd0260aa344dc72fe`. Zenodo declares
CC-BY-4.0 for the record; the Stable Diffusion Open RAIL-M terms also apply to
the derived generator.

Download the base model after accepting its Hugging Face terms:

```bash
hf download CompVis/stable-diffusion-v1-4 \
  --local-dir "$POSTGD_WEIGHTS/stable-diffusion-v1-4"
```

Older Hugging Face installations may use `huggingface-cli download` instead of
`hf download`. A local generation run then uses:

```bash
python stage1/full_method/generate.py \
  --count 1 \
  --base-model "$POSTGD_WEIGHTS/stable-diffusion-v1-4" \
  --weight-dir "$POSTGD_WEIGHTS/sd-retina-model" \
  --output-dir "$POSTGD_ROOT/runs/stage1_smoke" \
  --no-prior
```

`--allow-download` is required if the base model is not already cached and the
script is allowed to access the network. Do not use `--allow-base-fallback` for
the formal run: it substitutes the base Stable Diffusion UNet for the retina
fine-tuned UNet.

## 3. Vessel segmentation prior

The formal 20k server run used the binary LWNet `wnet` checkpoint trained on
DRIVE. It is directly downloadable from the upstream repository:

```bash
mkdir -p "$POSTGD_WEIGHTS/wnet_drive"
curl -L --fail \
  https://cdn.jsdelivr.net/gh/agaldran/lwnet@ce72ddabcf5fc9af14197ac0d43bd29000b17df2/experiments/wnet_drive/model_checkpoint.pth \
  -o "$POSTGD_WEIGHTS/wnet_drive/model_checkpoint.pth"
sha256sum "$POSTGD_WEIGHTS/wnet_drive/model_checkpoint.pth"
```

The [pinned file on GitHub](https://github.com/agaldran/lwnet/blob/ce72ddabcf5fc9af14197ac0d43bd29000b17df2/experiments/wnet_drive/model_checkpoint.pth)
is the source reference. The download command uses jsDelivr because some
networks cannot reach `raw.githubusercontent.com` reliably.

Expected SHA-256:

```text
91f0cada4b26ece63464b05be60f9b3a51f1bcd1f764081d07d3a35961118de1
```

Pass the **directory**, not the file, to the public adapter:

```bash
--vessel-model "$POSTGD_WEIGHTS/wnet_drive"
```

The adapter then looks for `model_checkpoint.pth` in that directory. It also
imports `engine.models.get_model` and `engine.utils.model_saving_loading`; those
historical modules are not included in this minimal public checkout. Thus the
checkpoint is obtainable, but the download alone is not a complete runnable
vessel prior for this release. A `--no-prior` run deliberately disables vessel,
fovea, and optic-disc avoidance and may place spots over vessels.

The historical GUI used separate `big_wnet` DRIVE/HRF A/V checkpoints. They are
not substitutes for the formal `wnet_drive` file.

## 4. Fovea and optic-disc localization

The upstream Fundus Image Toolbox downloads its fovea/optic-disc weights from
[Zenodo record 11174642](https://zenodo.org/records/11174642). If the human
record page returns a temporary `504 Gateway Time-out`, the same record is
available as [Zenodo API metadata](https://zenodo.org/api/records/11174642), and
the direct file endpoint below does not require the HTML page. The checkpoint
used by the formal run was:

```text
2024-05-07 11_13.05/multi_efficientnet-b3_best.pt
```

Let the toolbox fetch the files into a local directory, or download
[`weights.tar.gz`](https://zenodo.org/api/records/11174642/files/weights.tar.gz/content)
and preserve the directory name above when extracting:

```bash
mkdir -p "$POSTGD_WEIGHTS/fovea_od"
python - <<'PY'
import fundus_image_toolbox as fit
fit.load_fovea_od_model(
    checkpoint_dir="models/fovea_od",
    device="cpu",
    return_test_dataloader=False,
)
PY
```

If the archive is extracted manually, point `--fovea-model-dir` at the directory
that directly contains `multi_efficientnet-b3_best.pt`, for example:

```bash
--fovea-model-dir "$POSTGD_WEIGHTS/fovea_od/2024-05-07 11_13.05"
```

The Zenodo record does not declare a license for this checkpoint. The software
repository is MIT, but that does not clear redistribution of the weights.

## 5. DINOv3 initialization

The distillation script creates `vit_base_patch16_dinov3` through `timm` and
loads a user-provided local checkpoint. DINOv3 files are gated by Meta; request
access and accept the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
before downloading.

Save the authorized checkpoint as:

```text
models/dinov3/vit_base_patch16_dinov3.pth
```

For the formal configuration, use the same initialization file for teacher and
student:

```bash
python distillation/distill.py \
  --data-dir "$POSTGD_ROOT/runs/stage1_20k" \
  --output-dir "$POSTGD_ROOT/runs/dinov3_distill_20k" \
  --teacher-model vit_base_patch16_dinov3 \
  --student-model vit_base_patch16_dinov3 \
  --teacher-weights "$POSTGD_WEIGHTS/dinov3/vit_base_patch16_dinov3.pth" \
  --student-weights "$POSTGD_WEIGHTS/dinov3/vit_base_patch16_dinov3.pth" \
  --no-pretrained \
  --epochs 200 \
  --batch-size 64 \
  --expected-images 20000 \
  --validation-fraction 0 \
  --seed 42
```

The command writes a raw `student_final.pth` state dict, plus the training
configuration and resumable checkpoint, to the output directory.

## 6. PostGD distilled student

The formal candidate is the file produced by the 20k, 200-epoch run:

```text
student_final.pth
size:   342,626,578 bytes (326.75 MiB)
sha256: 1e5843650fd94b5ab12ce298fb3a30ec550982efa6ad62dbed04697f0aca7e00
format: raw PyTorch state_dict() for vit_base_patch16_dinov3
```

The recorded training configuration is `20,000` generated images, `200` epochs,
batch size `64`, learning rate `1e-4`, weight decay `0.05`, `500` warmup steps,
`beta=0.1`, seed `42`, and `pretrained=true` for the DINOv3 initialization.

The matching local/server candidate is not hosted yet. Keep it at
`models/student/student_final.pth` when testing a clone. It is not a checkpoint
wrapper with a `student` or `state_dict` key; the loader receives the state dict
directly.

Minimal load check:

```python
from pathlib import Path

import timm
import torch

path = Path("models/student/student_final.pth")
model = timm.create_model("vit_base_patch16_dinov3", pretrained=False, num_classes=0)
state = torch.load(path, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()
print("loaded", path, "with", len(state), "tensors")
```

Do not publish the smoke, old DINOv2, old 50-epoch, or spatial-distillation
files under this name. They are different experiments.

## 7. Release checklist

The file is small enough for ordinary Google Drive sharing: 326.75 MiB before
any archive wrapper. Hugging Face is preferable for a public reproducibility
release because it provides versioning, a model card, and a stable download API.
Neither platform changes the license obligations.

Before adding a public URL to this README:

1. Confirm that redistribution of a DINOv3-derived student checkpoint is allowed
   by the DINOv3 license and any applicable upstream terms.
2. Upload only the formal `student_final.pth`, a sanitized copy of its
   `config.json`, and a checksum file. Remove server absolute paths and do not
   upload private images or experiment logs.
3. Verify the published file with the SHA-256 above, then add the permanent URL
   and the exact revision to this document and the root README.

## 8. Hugging Face release procedure

Create a new **Model** repository on Hugging Face, for example
`Deliver0/postgd-dinov3-student`. Upload only PostGD-owned release artifacts:

```text
student_final.pth       formal PostGD distilled student
config.json             sanitized training configuration
SHA256SUMS              checksum for student_final.pth
README.md               model card and upstream attribution
```

Do not upload the DINOv3 teacher initialization, Stable Diffusion base or
fine-tuned UNet, LWNet W-Net, Fundus Image Toolbox fovea/optic-disc checkpoint,
clinical images, or `last_checkpoint.pt`. Those are third-party or private
artifacts. The current code uses fovea (macula) and optic-disc localization;
there is no separate PostGD optic-cup model in this release.

After creating the repository in the Hugging Face web interface, upload the
prepared folder with the official CLI:

```bash
python -m pip install -U huggingface_hub
hf auth login
hf upload Deliver0/postgd-dinov3-student \
  ./postgd-student-release . \
  --repo-type model
```

The 326.75 MiB file is handled by Hugging Face storage without adding it to the
GitHub repository. Keep the repository private until the DINOv3 license review
is complete, then switch it to public and add the resulting URL to this guide.
The model card should state that `student_final.pth` is a raw
`vit_base_patch16_dinov3` state dict, include the training configuration and
SHA-256, link to the DINOv3 license, and say that the artifact is research-only
and not a medical device.

For a local verification:

```bash
sha256sum models/student/student_final.pth
```

On PowerShell, use:

```powershell
Get-FileHash .\models\student\student_final.pth -Algorithm SHA256
```
