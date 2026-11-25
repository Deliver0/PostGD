"""Export frozen student embeddings for support/query classification."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / 'segmentation'))
from few_shot_segmentation import load_backbone, sha256
from seg_head import preprocess, load_rgb


@torch.no_grad()
def embedding(backbone, image, device, size, mode):
    x, _ = preprocess(image, size=size, mode=mode)
    value = backbone.forward_features(x.unsqueeze(0).to(device))
    if isinstance(value, dict):
        value = next((value[key] for key in ('x_norm_clstoken', 'cls_token', 'last_hidden_state', 'x')
                      if key in value), None)
    if not torch.is_tensor(value):
        raise ValueError('expected tensor embeddings from the student backbone')
    if value.ndim == 3:
        value = value[:, 0]
    if value.ndim != 2 or value.shape[0] != 1:
        raise ValueError('expected a [1,D] student embedding')
    return value[0].float().cpu().numpy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--student-weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backbone', default='vit_base_patch16_dinov3')
    parser.add_argument('--image-size', type=int, default=224)
    parser.add_argument('--preprocess-mode', choices=['stretch', 'center_crop', 'letterbox'], default='stretch')
    parser.add_argument('--model-kwargs', default='{}')
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if args.output.suffix != '.npz':
        raise ValueError('--output must end with .npz')
    payload = json.loads(args.manifest.read_text(encoding='utf-8'))
    samples = {sample['image_id']: sample for sample in payload['samples']}
    support, query = payload['support'], payload['query']
    if len(samples) != len(payload['samples']):
        raise ValueError('sample IDs must be unique')
    if (not support or not query or set(support) & set(query)
            or len(set(support)) != len(support) or len(set(query)) != len(query)):
        raise ValueError('support/query IDs must be nonempty, unique and disjoint')
    device = torch.device(args.device)
    kwargs = json.loads(args.model_kwargs)
    backbone = load_backbone(args.backbone, args.student_weights, device, kwargs).eval()
    arrays = {}
    for split, ids in [('support', support), ('query', query)]:
        features, labels = [], []
        for image_id in ids:
            sample = samples[image_id]
            if sample['label'] not in (0, 1):
                raise ValueError('classification labels must be 0 or 1')
            image = Path(sample['image'])
            if not image.is_absolute():
                image = args.manifest.parent / image
            features.append(embedding(backbone, load_rgb(image), device, args.image_size, args.preprocess_mode))
            labels.append(sample['label'])
        arrays[f'{split}_features'] = np.stack(features)
        arrays[f'{split}_labels'] = np.asarray(labels, dtype=np.int64)
        arrays[f'{split}_ids'] = np.asarray(ids, dtype=str)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    provenance = {'student_sha256': sha256(args.student_weights),
                  'manifest_sha256': sha256(args.manifest), 'backbone': args.backbone,
                  'image_size': args.image_size, 'preprocess_mode': args.preprocess_mode,
                  'model_kwargs': kwargs, 'support_ids': support, 'query_ids': query}
    args.output.with_suffix('.provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    return 0


if __name__ == '__main__':
    main()
