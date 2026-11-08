I’ll now restore the even gains and increase only the odd gains by 50%. Together, the complementary probes recover the full last-write vector while each leaves half the coordinates untouched as anchors.
```python
# Hypothesis: complementary selective-gain interventions recover the complete last-write direction and enable a norm-aware causal repair.
probe_gain=gpost.clone()
probe_gain[odd_idx]*=(1+probe_delta)
save_edit(POST_NAME,probe_gain)
for group in ["fit_a","fit_b","dev"]:
    (WORK/f"probe_odd_{group}.json").write_text(json.dumps({"type":"read","model":"edited",
        "from":{"pass":1,"ids":read_groups[group]}}))
note("Pass 7 completed: even-gain dev probe, fwd_7.\n\nPlanned pass 8: READ edited fit_a. Exact edit replaced, not compounded: layers.23.post_ffn_norm odd coordinates x1.5; even coordinates original. IDs in probe_odd_fit_a.json. Together with pass 5 this identifies all final-write coordinates.")
print("Current edit:",edited(),"changed odd coordinates",len(odd_idx))
```
