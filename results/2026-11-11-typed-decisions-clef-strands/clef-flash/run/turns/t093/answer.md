The practice result exactly reproduces the earlier run: **+5.67 percentage points, paired SE 0.99 points**. I’ll record that result and verify the final tensor fingerprints before the last validation pass. No further fitting or tuning is planned.


```python
# Hypothesis: the restored best_2 edit exactly reproduces its prior paired-practice gain and is ready for final validation.
assert edit_fingerprint() == _final_fingerprint
assert set(edited()) == set(_edit_names)
assert all(torch.equal(load(n), v.reshape(SHAPES[n]).to(torch.bfloat16).float())
           for n, v in zip(_edit_names, _best2_tuple))

_final_practice_result = {
    "n": 300,
    "original": 0.7053333333333334,
    "edited": 0.762,
    "paired_gain": 0.056666666666666664,
    "paired_se": 0.009925399928010711,
    "improved": 104,
    "worsened": 39,
    "repeat_of_pass": 10,
}
_changed_counts = {n: int((load(n) != load_original(n)).sum()) for n in _edit_names}
print("Changed entries:", _changed_counts, "total", sum(_changed_counts.values()))
print("Final paired practice:", _final_practice_result)
_manifest_path = _handoff_work / "final_edit_manifest.json"
_manifest = json.loads(_manifest_path.read_text())
_manifest["changed_entries"] = _changed_counts
_manifest["practice_result"] = _final_practice_result
_manifest_path.write_text(json.dumps(_manifest, indent=2))

note("\n## Pass 24 final practice check\n"
     "Original0.705333, edited0.762000; paired gain+0.056667, SE0.009925, n300;104 improved/39 worsened. "
     "Exactly reproduces pass10; these are the same records, so n remains300. No program/metric failures. "
     "Saved candidate was fitted on500 practice records and selected using100 tune records; this300-record check was not used to fit it. "
     "Incremental extension benefit over shared-only remains uncertain (+0.000667 ±0.005129 on these records).\n")
note("\n## Planned pass 25 — final validation deployment confirmation\n"
     "Exact final edit unchanged best_2, source targeted_candidate.pt;24 scorer neurons in three tensors, "
     "fingerprints verified equal to final_edit_manifest.json. Changed-entry counts: " + json.dumps(_changed_counts) +
     ". DECIDE both on all300 hidden validation items, aggregate only. "
     "This repeats pass11 to verify deployment, not an additional independent sample; no subsequent fitting or tuning. "
     "Leave this best-confirmed candidate deployed if the expected deterministic result is reproduced.\n")
```
