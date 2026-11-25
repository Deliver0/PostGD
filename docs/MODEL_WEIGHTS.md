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
│   ├── vit_large_patch16_dinov3.pth
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

The formal 200k server run used the binary LWNet `wnet` checkpoint trained on
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

The adapter looks for `model_checkpoint.pth` in that directory and loads it
through the self-contained `src/wnet_vessel.py` implementation. The historical
`engine.models` and `engine.utils` modules are not required. A `--no-prior`
PRP run disables vessel, fovea and optic-disc avoidance.

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

The distillation script creates a `vit_large_patch16_dinov3` teacher and a
`vit_base_patch16_dinov3` student through `timm` and
loads a user-provided local checkpoint. DINOv3 files are gated by Meta; request
access and accept the [DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)
before downloading.

Save the authorized checkpoint as:

```text
models/dinov3/vit_large_patch16_dinov3.pth
models/dinov3/vit_base_patch16_dinov3.pth
```

Use state dicts compatible with the requested timm architectures. When using
timm's pretrained download, accept the upstream model terms first, then export
each architecture separately:

```python
import timm
import torch

for size in ("large", "base"):
    name = f"vit_{size}_patch16_dinov3"
    model = timm.create_model(name, pretrained=True, num_classes=0)
    torch.save(model.state_dict(), f"models/dinov3/{name}.pth")
```

Use separate initialization files matching the teacher and student architectures.
The example below uses the public single-process batch size of 64. It does not
reproduce the 640-image multi-GPU batch specified in the response:

```bash
python distillation/distill.py \
  --data-dir "$POSTGD_ROOT/runs/stage1_200k/distill_224" \
  --output-dir "$POSTGD_ROOT/runs/dinov3_distill_200k" \
  --teacher-model vit_large_patch16_dinov3 \
  --student-model vit_base_patch16_dinov3 \
  --teacher-weights "$POSTGD_WEIGHTS/dinov3/vit_large_patch16_dinov3.pth" \
  --student-weights "$POSTGD_WEIGHTS/dinov3/vit_base_patch16_dinov3.pth" \
  --no-pretrained \
  --epochs 200 \
  --batch-size 64 \
  --expected-images 200000 \
  --validation-fraction 0 \
  --seed 42
```

The command writes a raw `student_final.pth` state dict, plus the training
configuration and resumable checkpoint, to the output directory.

## 6. PostGD distilled student

The pretrained student is available at
[Deliver0/postgd-dinov3-student](https://huggingface.co/Deliver0/postgd-dinov3-student).
It contains the backbone only; train task-specific heads using the downstream
guides. The model card describes its associated training configuration,
provenance and limitations.

Download the original PyTorch state dict for the PostGD scripts:

```bash
hf download Deliver0/postgd-dinov3-student student_final.pth \
  --revision ed5f107e49b6624eb9cdf206df257deffd3fe956 \
  --local-dir "$POSTGD_WEIGHTS/student"
```

Release details:

```text
student_final.pth
size:   342,626,578 bytes (326.75 MiB)
sha256: 3de942cfab4657d05c201207b4760ce07cd1537a49bccbb7565d930ed04d8544
format: raw PyTorch state_dict() for vit_base_patch16_dinov3
revision: ed5f107e49b6624eb9cdf206df257deffd3fe956
```

The model repository also includes an equivalent `model.safetensors`, inference
`config.json`, sanitized `training_config.json`, `SHA256SUMS` and the DINOv3
license. These derived weights use the DINOv3 license, not the MIT license for
the original PostGD code. The associated configuration records `beta=0.01`;
the manuscript recipe specifies `beta=0.1`. This artifact is not presented as a
verified reproduction of every manuscript setting.

For image embeddings, load the Safetensors version directly with `timm>=1.0.31`:

```python
import timm

model = timm.create_model("hf-hub:Deliver0/postgd-dinov3-student", pretrained=True)
model.eval()
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
```

Use RGB images with the returned transform; the default input is 224 × 224 and
the output is a 768-dimensional embedding. See the model card for a complete
inference example. For a local checksum check:

```bash
sha256sum models/student/student_final.pth
```

On PowerShell, use:

```powershell
Get-FileHash .\models\student\student_final.pth -Algorithm SHA256
```
