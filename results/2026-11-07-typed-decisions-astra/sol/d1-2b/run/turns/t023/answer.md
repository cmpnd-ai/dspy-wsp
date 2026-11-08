I’ll apply the first diagnostic gain pattern to the additional records. Reusing the same pattern makes the branch measurements comparable across both batches.

```python
# Hypothesis: causal branch sensitivity generalizes across fit records, not just the first batch.
save_edit(gn,g0+d1)
req2["model"]="edited"
json.dump(req2,open(work/"read_more_probe.json","w"))
note("Pass9 READ additional 200 fit cases with causal probe1 g0+d1; no other edits.")
```
