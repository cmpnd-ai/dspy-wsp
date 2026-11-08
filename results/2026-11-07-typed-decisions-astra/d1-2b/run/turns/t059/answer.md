I’ll check that the unchanged coordinates behave as predicted, then switch to the complementary odd-coordinate probe. That second read will complete the recovery of the final-write vectors.
```python
# Hypothesis: unchanged-coordinate consistency confirms that the gain probe isolates the last residual write.
extra_even = read_arrays(22)
extra_ce, extra_scale_e, extra_anchor_e = infer_subset(extra_orig, extra_even, even_idx)
print("Even probe anchor RMS error:", float(extra_anchor_e))
print("Final RMS scale ratio median:", float(np.median(extra_scale_e)))
probe_odd_gain = gpost.clone()
probe_odd_gain[odd_idx] *= 1.5
save_edit(post_name, probe_odd_gain)
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPass 22 expansion even-coordinate probe completed. Untouched-coordinate anchor RMS error %.3g; final-scale ratio median %.6f.\n" % (float(extra_anchor_e),float(np.median(extra_scale_e))))
    f.write("\nPlanned pass 23: READ edited expansion180, same probe_expansion180.json. Exact edit replaced with layers.23.post_ffn_norm.weight odd coordinates x1.5, even original; W2 and all other tensors original. Diagnostic only; norm edit will be removed after this read.\n")
```