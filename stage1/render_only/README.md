# Standalone spot renderer

This renderer accepts an RGB fundus background and writes a synthetic image,
hard spot mask, metadata, and optional diagnostic prior masks.

```bash
python render_spots.py \
  --input /path/to/background.png \
  --output-dir ./output \
  --spot-size 2 \
  --gap 0.001 \
  --num-spots 160 \
  --dark-center 0 \
  --dark-to-bright 0 \
  --bright-zone 0.2 \
  --fade-zone 0.8 \
  --dark-depth 0 \
  --gain 0.12 \
  --no-prior
```

Anatomy-aware placement is enabled by default when the caller provides external
prior checkpoints. A missing vessel prediction blocks placement rather than
silently allowing a spot to cover a vessel.

