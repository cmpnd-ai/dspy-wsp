The cached ablations suggest that the low-energy dimensions account for nearly all the gain. I’ll verify this through the actual model, starting with an edit restricted to the ten dominant dimensions. This ablation is diagnostic, not a replacement for the validated repair.


```python
# Hypothesis: changing only the dominant activation dimensions will not reproduce the repair's accuracy gain.
for fn in (correction, accuracy_stats, x_for, center_options):
    try:
        print(inspect.getsource(fn))
    except (OSError, TypeError):
        print(fn.__name__, "arguments", inspect.signature(fn), "constants", fn.__code__.co_consts[:12])
delta_top10 = torch.outer(u, v_outlier)
save_edit("k.weight", K0 + delta_top10)
torch.save({"u":u, "v_full":v_ref, "v_top10":v_outlier, "v_other2038":v_lowenergy,
            "v_schema_mean":v_mean_span, "v_schema_null":v_schema_null,
            "top10_dims":top10, "schema_rank":mu_rank}, WORK/"ablation_directions.pt")
print("Existing paired development request:", json.load(open(WORK/"paired_dev.json")))
with open(WORK/"notes.md","a") as f:
    f.write("\nCaution: correction() includes option-common offsets; RMS values printed in run28 were NOT centered. "
            "Only accuracy and paired gain from that run are used; Brier must respect per-type temperature.\n")
    f.write("\n## Before pass 16\nExact edit: k.weight = K0 + outer(u,v_top10), "
            "v_top10 retains only dims [98,1816,1035,893,1855,281,1630,2,636,456] of validated v_ref; "
            "all other tensors original. Directions in ablation_directions.pt. Paired DECIDE same 180 development cases "
            "in paired_dev.json. Causal ablation only; best2 remains the saved validated choice. "
            "Cached prediction: accuracy .577778, gain 0.\n")
```
