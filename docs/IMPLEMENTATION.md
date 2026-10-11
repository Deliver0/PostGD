# Implementation contract

This document records the public implementation contract. It intentionally uses
repository-relative paths and external placeholders instead of private server
paths.

## Stage 1

The complete generator is `stage1/full_method/generate.py`. Its order is:

1. obtain a background from Stable Diffusion or `--background-dir`;
2. apply a Lab profile to the non-black FOV;
3. build optional vessel, optic-disc, fovea, and posterior-pole priors;
4. place spots on concentric rings around the posterior pole;
5. render an irregular footprint with a continuous radial intensity profile;
6. crop the FOV to a centred circle and resize to the distillation input size;
7. save the final image, masks, metadata, and manifest.

Placement is a radial PRP walk, not uniform sampling. Ring radii are authored at
the 518x395 clinical planning size (`reference_base_radius=55`,
`reference_ring_step=20`, `rings=10`) and rescaled to the working resolution, so
the layout is identical at 512x512 and at a native 2300x1830 fundus. The walk
starts outside the posterior-pole exclusion, which is centred on the optic disc
following the ETDRS convention, and alternates a half-slot angular offset
between neighbouring rings. Every centre is jittered by up to `angular_jitter`
radians and `radial_jitter` ring steps.

The requested spot count is an upper bound, never a target. Each candidate is
rejected when it falls outside the FOV, inside the posterior-pole exclusion, on
the dilated vessel/optic-disc avoidance mask, or closer than the minimum centre
distance to an accepted spot. The actual number of burns is recorded in the
metadata and manifest.

The ring profile is a continuous peak, not a constant-intensity annulus:

```text
background -> inner shoulder -> local peak -> outer shoulder -> background
```

The `bright_no_dark` profile sets `dark_center=0`, `dark_to_bright=0`, and
`dark_depth=0`, so it does not create an artificial dark center. It is the
default appearance of both entry points: `stage1/full_method/generate.py`
(`--spot-case bright_no_dark`) and the standalone `render_spots.py` CLI, whose
profile defaults are solid for the same reason.

Two approved spot profiles ship with the generator. `generate.py` selects one
with `--spot-case`:

```text
bright_no_dark:
  spot_size=1.3, gap=0.001, gain=0.12
  dark_center=0, dark_to_bright=0, bright_zone=0.20
  fade_zone=0.80, dark_depth=0

dark_center:
  spot_size=1.8, gap=0.02, gain=0.10
  ring_center=0.55, ring_width=0.05
  ring_inner_fade=0.25, ring_outer_fade=0.45
  ring_shoulder_power=1.0, dark_depth=0.04
```

Spot brightness follows `region * (1 + gain * m * warm)`, where `m` is the
radial profile and `warm` is a per-channel tint. `warm` defaults to the orange
`(1.80, 0.85, 0.40)` validated against real post-operative fundus images; a
near-neutral tint under-represents the warm laser-scar appearance. At
`gain=0.18` a bright region saturates the red channel, producing the warm-white
burn seen clinically.

## Stage 2

`distillation/distill.py` uses two independently augmented views. The teacher is
frozen and the student is optimized with:

```text
L_total = L_KD + beta * L_cons
L_KD     = MSE(projector(student(view_1)), teacher(view_1))
L_cons   = 1 - cosine(student(view_1), student(view_2))
```

The teacher is `vit_large_patch16_dinov3` (1024-d) and the student is
`vit_base_patch16_dinov3` (768-d), connected by a `Linear(768, 1024)`
projection. **Both are initialised from pretrained DINOv3 weights.** A randomly
initialised student collapses under this objective: with only the 200k synthetic
images and no natural-image pretraining the student features collapse onto a
single direction (pairwise cosine ~0.99, effective rank 13/384), which then
scores at chance on every downstream task.

The default formal run uses 200,000 images, 200 epochs, global batch size 640,
AdamW, learning rate `2e-4`, weight decay `0.05`, 500 warmup steps, cosine
decay, layer-wise learning-rate decay `0.75`, an EMA copy of the student with
`ema_decay=0.999`, and `beta=0.1`.

## Stage 3

`downstream/adapt_head.py` accepts frozen features and a support/query split.
Candidate selection uses only support labels. The query labels are used only after
the candidate is locked and are never used for model selection.

The public script includes:

- raw cosine prototypes;
- weak-augmentation or view-averaged prototypes when supplied;
- shrinkage prototypes;
- a learnable prototype head;
- L2-normalized logistic regression.

## External prerequisites

The code does not redistribute:

- clinical data;
- generated image datasets;
- foundation-model weights;
- local vessel or fovea/optic-disc checkpoints.

When anatomy priors are enabled, the caller supplies the prior model locations.
The renderer can be run in an explicit `--no-prior` mode for software tests and
non-clinical examples.

