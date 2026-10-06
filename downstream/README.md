# Few-shot downstream head

`adapt_head.py` implements the released support-only head selection protocol on
frozen feature arrays. It does not assume a particular hospital directory layout.

```bash
python downstream/adapt_head.py \
  --support-npz /path/to/support_and_query.npz \
  --output ./runs/downstream_result.json
```

The NPZ keys are `support_features`, `support_labels`, `query_features`, and
`query_labels`. The query labels are withheld until the final metric calculation.

