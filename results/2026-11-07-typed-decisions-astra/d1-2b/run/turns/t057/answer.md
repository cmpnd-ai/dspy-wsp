I’ll temporarily remove the weight repair and multiply only the even-coordinate gains of the last post-FFN norm by 1.5. The untouched odd coordinates will let me account for the final RMS normalization and isolate the even-coordinate write contribution. This is a diagnostic probe, not a candidate repair.
```python
# Hypothesis: complementary coordinate-gain probes recover the final MLP write without executing the backbone locally.
extra_orig = read_arrays(21)
post_name = "model.language_model.layers.23.post_ffn_norm.weight"
drop_edit(w2_name)
probe_even_gain = gpost.clone()
probe_even_gain[even_idx] *= 1.5
save_edit(post_name, probe_even_gain)
extra_probe_request = {
    "type":"read", "model":"edited",
    "from":{"pass":1,"ids":expand_ids},
    "layers":[23,24]
}
json.dump(extra_probe_request, (work_dir/"probe_expansion180.json").open("w"))
print("Extra original questions:",len(extra_orig["h"]))
print("Probe delta:",probe_delta)
print("infer_subset return instructions:",[(i.opname,i.argrepr) for i in list(dis.get_instructions(infer_subset))[-18:]])
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPass 21 completed: original readout for expansion180 (900 questions), fwd_21.\n")
    f.write("\nPlanned pass 22: READ edited expansion180, probe_expansion180.json. Exact edit: layers.23.post_ffn_norm.weight even coordinates x1.5, odd original. W2 repair removed. All other weights original. Diagnostic only; confirmed best_expanded540 preserved.\n")
```