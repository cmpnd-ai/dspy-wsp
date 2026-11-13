I now have the bundle layout. I’ll verify the saved tensors and compute each repair’s logit changes from the recorded scorer inputs. Because these edits touch only the final scorer, this calculation can isolate their effect without rerunning or approximating the backbone.


```python
# Hypothesis: final-scorer edits reproduce the prespecified cue-margin effects on independent seeds.
_edit_names = ["residual_scorer.0.weight", "residual_scorer.0.bias", "residual_scorer.3.weight"]
_best1_tuple = (_saved_best1["first_weight"], _saved_best1["first_bias"], _saved_best1["last_weight"])
_best2_tuple = _saved_best2["tensors"]
_integrity = {
    n: bool(torch.equal(load(n), v.reshape(SHAPES[n]).to(torch.bfloat16).float()))
    for n, v in zip(_edit_names, _best2_tuple)
}
print("Exact best_2/edit equality:", _integrity)
assert all(_integrity.values())

_rows22 = _meta22["rows"]
_features22 = torch.from_numpy(np.load(_handoff_work / "fwd_22/head_inputs.npy")).nan_to_num()
_mask22 = torch.from_numpy(np.load(_handoff_work / "fwd_22/option_mask.npy")).bool()
_original22 = torch.from_numpy(np.load(_handoff_work / "fwd_22/option_logits.npy"))
_shared22 = simulate_probe_scoring(_features22, _mask22, _original22, _best1_tuple)
_full22 = simulate_probe_scoring(_features22, _mask22, _original22, _best2_tuple)
print("Simulated arrays:", tuple(_shared22.shape), tuple(_full22.shape))
print("Probe families:")
for _fam in sorted({m["family"] for m in held_probe_meta}):
    print(_fam, sorted({str(m["level"]) for m in held_probe_meta if m["family"] == _fam}))
print("Unique customer action options:", next(r["options"] for r in _rows22 if r["question"] == "action" and "answer_directly" in r["options"]))
print("Edit fingerprint:", edit_fingerprint())
```
