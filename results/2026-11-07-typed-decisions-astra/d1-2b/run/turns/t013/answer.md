I’ll now increase only the even-indexed gains in the last post-FFN norm by 50%. This is a diagnostic intervention, not a candidate final edit: if the last write carries decision-relevant evidence, its option contrasts should change, and the untouched odd coordinates will let me separate that change from final RMS normalization.
```python
# Hypothesis: the last FFN residual write causally controls the mistaken option contrasts and can be isolated using untouched coordinates.
probe_delta=0.5
even_idx=np.arange(0,2048,2)
odd_idx=np.arange(1,2048,2)
probe_gain=gpost.clone()
probe_gain[even_idx]*=(1+probe_delta)
save_edit(POST_NAME,probe_gain)
for group in ["fit_a","fit_b","dev"]:
    (WORK/f"probe_even_{group}.json").write_text(json.dumps({"type":"read","model":"edited",
        "from":{"pass":1,"ids":read_groups[group]}}))
note("Planned pass 5: READ edited fit_a (same 180 IDs as original pass 2). Exact edit: layers.23.post_ffn_norm.weight even coordinates multiplied by 1.5, odd coordinates original; all other tensors original. Diagnostic only, not a proposed final edit. Untouched odd coordinates anchor the final-RMS ratio; even-coordinate differences isolate the corresponding last-write contributions.")
print("Current edit:",edited(),"changed coordinates",len(even_idx))
```
