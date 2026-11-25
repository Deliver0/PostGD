"""Image-based segmentation with task-specific adaptation of the final ViT blocks."""
from __future__ import annotations

import copy
import json
import random

import torch
from torch import nn
from PIL import ImageOps

from seg_head import (
    DEFAULT_FEATURE_INDICES, SegDecoder, binary_metrics, estimate_fg_weight,
    load_mask, load_rgb, paired_augment, preprocess, seed_everything,
    segmentation_loss,
)


def task_finetune_blocks(task: str) -> int:
    """Default adaptation budget; other segmentation tasks keep frozen features."""
    key = task.lower().replace('-', '_').replace(' ', '_')
    if key in {'od', 'disc', 'optic_disc', 'optic_disc_segmentation'}:
        return 1
    if key in {'oc', 'cup', 'optic_cup', 'optic_cup_segmentation'}:
        return 2
    return 0


class PartialSegmentationModel(nn.Module):
    """Own an independent backbone copy and train only its final N blocks + decoder."""

    def __init__(self, backbone, blocks: int, feature_indices=DEFAULT_FEATURE_INDICES):
        super().__init__()
        self.backbone = copy.deepcopy(backbone)
        sequence = getattr(self.backbone, 'blocks', None)
        if sequence is None or not 1 <= blocks <= len(sequence):
            raise ValueError('finetune blocks must be between 1 and the backbone block count')
        self.block_count = blocks
        self.feature_indices = tuple(feature_indices)
        if max(self.feature_indices) != len(sequence) - 1:
            raise ValueError('the decoder must consume the final block to adapt it')
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        for block in sequence[-blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad_(True)
        self.decoder = SegDecoder(embed_dim=int(self.backbone.embed_dim))

    def train(self, mode=True):
        super().train(mode)
        # Frozen blocks retain evaluation behavior (including dropout/stochastic depth).
        self.backbone.eval()
        if mode:
            for block in self.backbone.blocks[-self.block_count:]:
                block.train()
        return self

    def forward(self, images):
        # No no_grad/detach/cache here: gradients must reach the selected blocks.
        features = self.backbone.forward_intermediates(
            images, indices=self.feature_indices, return_prefix_tokens=False,
            norm=False, output_fmt='NCHW', intermediates_only=True,
        )
        return self.decoder(features)

    def parameter_groups(self, decoder_lr, backbone_lr):
        return [
            {'params': self.decoder.parameters(), 'lr': decoder_lr},
            {'params': [p for p in self.backbone.parameters() if p.requires_grad],
             'lr': backbone_lr},
        ]


def support_tensors(samples, task, ids, image_size, mode, aug_repeats, seed):
    """Prepare image tensors, never cached backbone features or test annotations."""
    rng = random.Random(seed)
    original, augmented = {}, {}
    for image_id in ids:
        sample = samples[image_id]
        image, mask = load_rgb(sample.image), load_mask(sample.masks[task])
        original[image_id] = preprocess(image, mask, size=image_size, mode=mode)
        augmented[image_id] = [
            preprocess(*paired_augment(image, mask, rng, index), size=image_size, mode=mode)
            for index in range(aug_repeats)
        ]
    return original, augmented


def fit_partial(backbone, blocks, ids, original, augmented, device, epochs,
                decoder_lr, backbone_lr, seed, dice_weight=0.5, loss_mode='foreground'):
    if not ids:
        raise ValueError('at least one training image is required')
    seed_everything(seed)
    model = PartialSegmentationModel(backbone, blocks).to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(decoder_lr, backbone_lr), weight_decay=1e-4,
    )
    masks = {image_id: original[image_id][1] for image_id in ids}
    fg_weight = estimate_fg_weight(ids, masks)
    items = [item for image_id in ids
             for item in [original[image_id], *augmented[image_id]]]
    model.train()
    for _ in range(epochs):
        random.shuffle(items)
        for image, mask in items:
            optimizer.zero_grad(set_to_none=True)
            logits = model(image.unsqueeze(0).to(device))
            loss = segmentation_loss(logits, mask.unsqueeze(0).to(device),
                                     fg_weight, dice_weight, loss_mode)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 5.0,
            )
            optimizer.step()
    return model


@torch.no_grad()
def probability(model, image, device, image_size, mode, tta=False):
    model.eval()
    x, _ = preprocess(image, size=image_size, mode=mode)
    probabilities = [model(x.unsqueeze(0).to(device)).softmax(1)[0, 1]]
    if tta:
        x, _ = preprocess(ImageOps.mirror(image), size=image_size, mode=mode)
        probabilities.append(model(x.unsqueeze(0).to(device)).softmax(1)[0, 1].flip(-1))
    return torch.stack(probabilities).mean(0).cpu()


def select_partial(backbone, blocks, samples, support, original, augmented,
                   device, epochs, thresholds, decoder_lr, backbone_lr,
                   seed, image_size, mode, tta=False, dice_weight=0.5, loss_mode='foreground'):
    """Each support-only fold starts from the same original student checkpoint."""
    if len(support) < 2:
        return epochs[-1], 0.5, {}
    scores = {}
    for count in epochs:
        totals = [0.0] * len(thresholds)
        for fold, held_out in enumerate(support):
            fit_ids = [image_id for image_id in support if image_id != held_out]
            model = fit_partial(
                backbone, blocks, fit_ids, original, augmented, device, count,
                decoder_lr, backbone_lr, seed + 1000 * fold + count,
                dice_weight, loss_mode,
            )
            probs = probability(model, load_rgb(samples[held_out].image), device,
                                image_size, mode, tta)
            for j, threshold in enumerate(thresholds):
                metrics = binary_metrics(probs.ge(threshold), original[held_out][1])
                totals[j] += (metrics['dice'] + metrics['miou']) / 2
            del model
        scores[count] = [value / len(support) for value in totals]
    best_epoch, best_j = max(
        ((epoch, j) for epoch in scores for j in range(len(thresholds))),
        key=lambda item: scores[item[0]][item[1]],
    )
    return best_epoch, float(thresholds[best_j]), scores


def run_partial_task(backbone, blocks, task, samples, support, test, args, device):
    """Select using support only, refit, then open held-out masks for final scoring."""
    if set(support) & set(test):
        raise ValueError('support and test image IDs must be disjoint')
    if args.lr <= 0 or args.backbone_lr <= 0:
        raise ValueError('decoder and backbone learning rates must be positive')
    if not args.epoch_candidates or min(args.epoch_candidates) < 1 or not args.threshold_candidates:
        raise ValueError('provide positive epoch candidates and at least one threshold')
    original, augmented = support_tensors(
        samples, task, support, args.image_size, args.preprocess_mode,
        args.aug_repeats, args.seed,
    )
    best_epoch, threshold, cv = select_partial(
        backbone, blocks, samples, support, original, augmented, device,
        args.epoch_candidates, args.threshold_candidates, args.lr, args.backbone_lr,
        args.seed, args.image_size, args.preprocess_mode, args.tta,
        args.dice_weight, args.loss_mode,
    )
    model = fit_partial(
        backbone, blocks, support, original, augmented, device, best_epoch,
        args.lr, args.backbone_lr, args.seed, args.dice_weight, args.loss_mode,
    )
    rows = []
    for image_id in test:
        sample = samples[image_id]
        probs = probability(model, load_rgb(sample.image), device, args.image_size,
                            args.preprocess_mode, args.tta)
        _, mask = preprocess(load_rgb(sample.image), load_mask(sample.masks[task]),
                             size=args.image_size, mode=args.preprocess_mode)
        rows.append({'id': image_id, **binary_metrics(probs.ge(threshold), mask)})
    task_dir = args.output / task
    task_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = task_dir / 'task_model.pth'
    torch.save({
        'backbone': {k: v.cpu() for k, v in model.backbone.state_dict().items()},
        'decoder': {k: v.cpu() for k, v in model.decoder.state_dict().items()},
        'task': task, 'backbone_name': args.backbone,
        'finetune_blocks': blocks, 'decoder_lr': args.lr, 'backbone_lr': args.backbone_lr,
        'image_size': args.image_size, 'preprocess_mode': args.preprocess_mode,
        'model_kwargs': json.loads(args.model_kwargs),
        'feature_indices': list(DEFAULT_FEATURE_INDICES),
        'selected_epoch': best_epoch, 'selected_threshold': threshold, 'tta': args.tta,
    }, checkpoint)
    return rows, {
        'selected_epoch': best_epoch, 'selected_threshold': threshold,
        'support_cv': {str(k): v for k, v in cv.items()},
        'n_support': len(support), 'n_test': len(test),
        'finetune_blocks': blocks, 'decoder_lr': args.lr,
        'backbone_lr': args.backbone_lr, 'checkpoint': str(checkpoint),
    }
