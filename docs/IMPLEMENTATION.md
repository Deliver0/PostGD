# Implementation contract

This document records the public implementation contract. It intentionally uses
repository-relative paths and external placeholders instead of private server
paths.

## Stage 1

The complete generator is `stage1/full_method/generate.py`. Its order is:

1. obtain a background from Stable Diffusion or `--background-dir`;
2. apply a Lab profile to the non-black FOV;
3. build optional vessel, optic-disc, fovea, and posterior-pole priors;
4. place non-overlapping spots inside the FOV;
5. render an irregular footprint with a continuous radial intensity profile;
6. save the final image, masks, metadata, and manifest.

The spot count is an upper bound. The placement code first checks FOV membership,
central exclusion, prior avoidance, and pairwise minimum distance. It then
records the actual number in metadata.

The ring profile is a continuous peak, not a constant-intensity annulus:

```text
background -> inner shoulder -> local peak -> outer shoulder -> background
```

The `bright_no_dark` profile sets `dark_center=0`, `dark_to_bright=0`, and
`dark_depth=0`, so it does not create an artificial dark center.

## Stage 2

`distillation/distill.py` uses two independently augmented views. The teacher is
frozen and the student is optimized with:

```text
L_total = L_KD + beta * L_cons
L_KD     = MSE(projector(student(view_1)), teacher(view_1))
L_cons   = 1 - cosine(student(view_1), student(view_2))
```

The default formal run uses 20,000 images, 200 epochs, batch size 64, AdamW,
learning rate `1e-4`, weight decay `0.05`, 500 warmup steps, cosine decay, and
`beta=0.1`.

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

