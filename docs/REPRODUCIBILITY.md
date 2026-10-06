# Server-run reproducibility checklist

The original experiments were executed on a server. This checklist keeps the
public repository independent from server-specific paths while preserving the
experiment contract.

## Before stage 1

- [ ] Install `stage1/full_method/requirements.txt`.
- [ ] Make Stable Diffusion components and the fine-tuned UNet available locally.
- [ ] Make the vessel and fovea/optic-disc prior checkpoints available, or
      explicitly use `--no-prior`.
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

