I’ll check that the edited read matches the prediction and leaves upstream inputs unchanged. Then I’ll prepare the final paired practice check with the same saved weights; the repeat is a deployment check, not additional statistical evidence.


```python
# Hypothesis: the repaired logits are caused solely by the selected scorer neurons, with upstream routing unchanged.
_meta23 = json.loads((_handoff_work / "fwd_23/meta.json").read_text())
assert _meta23["rows"] == _meta22["rows"]
_actual23 = torch.from_numpy(np.load(_handoff_work / "fwd_23/option_logits.npy"))
_features23 = torch.from_numpy(np.load(_handoff_work / "fwd_23/head_inputs.npy")).nan_to_num()
_feature_error23 = float((_features23 - _features22).abs().max())
_logit_error23 = float((_actual23[_mask22] - _full22[_mask22]).abs().max())
_upstream_errors23 = {}
for _name in ("prior", "cosine"):
    _a22 = np.load(_handoff_work / f"fwd_22/{_name}.npy")
    _a23 = np.load(_handoff_work / f"fwd_23/{_name}.npy")
    _upstream_errors23[_name] = float(np.nanmax(np.abs(_a23 - _a22)))
print("Max errors:", {"features": _feature_error23, "predicted_logits": _logit_error23, **_upstream_errors23})
assert _feature_error23 == 0
assert _logit_error23 < 0.002
assert all(v == 0 for v in _upstream_errors23.values())

_req24 = json.loads((_handoff_work / "test_linear_bypass.json").read_text())
print("Reserved practice request:", {k: len(v) if k == "problems" else v for k, v in _req24.items()})
assert _req24["type"] == "decide" and _req24["model"] == "both" and len(_req24["problems"]) == 300
_final_fit_ids = set(fit_ids)
_final_tune_ids = set(tune_ids)
assert not (set(_req24["problems"]) & (_final_fit_ids | _final_tune_ids))
assert set(edited()) == set(_edit_names)
assert all(torch.equal(load(n), v.reshape(SHAPES[n]).to(torch.bfloat16).float())
           for n, v in zip(_edit_names, _best2_tuple))
_req24_path = _handoff_work / "final_practice_check.json"
_req24_path.write_text(json.dumps(_req24))
_final_fingerprint = edit_fingerprint()
(_handoff_work / "final_edit_manifest.json").write_text(json.dumps({
    "candidate": "best_2",
    "source": "targeted_candidate.pt",
    "neurons": _saved_best2["neurons"],
    "targeted_fields": _saved_best2["qkeys"],
    "fingerprint": _final_fingerprint,
    "fit_n": len(_final_fit_ids),
    "tune_n": len(_final_tune_ids),
    "reserved_test_n": 300,
}, indent=2))

note("\n## Pass 23 independent-seed causal verification\n"
     f"Actual best_2 read: max upstream feature difference {_feature_error23:g}; max simulated-vs-tool logit error {_logit_error23:g}; "
     f"prior/cosine differences {_upstream_errors23}. Tool errors1: partial−success original−0.346/full+0.389; "
     "violations0/1/2: critical−routine original[−2.137,−2.063,−1.699], full[−2.765,+0.028,+0.733]. "
     "Benign security context response strengthens; duplicate evidence still fails to overcome negative duplicate margin; customer review remains mixed.\n")
note("\n## Planned pass 24 — final reserved-practice deployment check\n"
     "Exact edit unchanged best_2 from targeted_candidate.pt, 24 scorer neurons; hashes in final_edit_manifest.json: "
     + json.dumps(_final_fingerprint) +
     ". DECIDE both on300 fixed reserved IDs in final_practice_check.json (same IDs as passes5/10), disjoint from500 fit and100 tune for this saved candidate. "
     "This deterministic repeat is an integrity check, NOT additional independent evidence; do not combine sample sizes. "
     "Rejected full-practice refits are not deployed. Final pass25 will repeat hidden validation for deployment confirmation only.\n")
print("Ready:", str(_req24_path))
```

```python
# Hypothesis: the repaired logits are caused solely by the selected scorer neurons, with upstream routing unchanged.
_meta23 = json.loads((_handoff_work / "fwd_23/meta.json").read_text())
assert _meta23["rows"] == _meta22["rows"]
_actual23 = torch.from_numpy(np.load(_handoff_work / "fwd_23/option_logits.npy"))
_features23 = torch.from_numpy(np.load(_handoff_work / "fwd_23/head_inputs.npy")).nan_to_num()
_feature_error23 = float((_features23 - _features22).abs().max())
_logit_error23 = float((_actual23[_mask22] - _full22[_mask22]).abs().max())
_upstream_errors23 = {}
for _name in ("prior", "cosine"):
    _a22 = np.load(_handoff_work / f"fwd_22/{_name}.npy")
    _a23 = np.load(_handoff_work / f"fwd_23/{_name}.npy")
    _upstream_errors23[_name] = float(np.nanmax(np.abs(_a23 - _a22)))
print("Max errors:", {"features": _feature_error23, "predicted_logits": _logit_error23, **_upstream_errors23})
assert _feature_error23 == 0
assert _logit_error23 < 0.002
assert all(v == 0 for v in _upstream_errors23.values())

_req24 = json.loads((_handoff_work / "test_linear_bypass.json").read_text())
print("Reserved practice request:", {k: len(v) if k == "problems" else v for k, v in _req24.items()})
assert _req24["type"] == "decide" and _req24["model"] == "both" and len(_req24["problems"]) == 300
_final_fit_ids = set(fit_ids)
_final_tune_ids = set(tune_ids)
assert not (set(_req24["problems"]) & (_final_fit_ids | _final_tune_ids))
assert set(edited()) == set(_edit_names)
assert all(torch.equal(load(n), v.reshape(SHAPES[n]).to(torch.bfloat16).float())
           for n, v in zip(_edit_names, _best2_tuple))
_req24_path = _handoff_work / "final_practice_check.json"
_req24_path.write_text(json.dumps(_req24))
_final_fingerprint = edit_fingerprint()
(_handoff_work / "final_edit_manifest.json").write_text(json.dumps({
    "candidate": "best_2",
    "source": "targeted_candidate.pt",
    "neurons": _saved_best2["neurons"],
    "targeted_fields": _saved_best2["qkeys"],
    "fingerprint": _final_fingerprint,
    "fit_n": len(_final_fit_ids),
    "tune_n": len(_final_tune_ids),
    "reserved_test_n": 300,
}, indent=2))

note("\n## Pass 23 independent-seed causal verification\n"
     f"Actual best_2 read: max upstream feature difference {_feature_error23:g}; max simulated-vs-tool logit error {_logit_error23:g}; "
     f"prior/cosine differences {_upstream_errors23}. Tool errors1: partial−success original−0.346/full+0.389; "
     "violations0/1/2: critical−routine original[−2.137,−2.063,−1.699], full[−2.765,+0.028,+0.733]. "
     "Benign security context response strengthens; duplicate evidence still fails to overcome negative duplicate margin; customer review remains mixed.\n")
note("\n## Planned pass 24 — final reserved-practice deployment check\n"
     "Exact edit unchanged best_2 from targeted_candidate.pt, 24 scorer neurons; hashes in final_edit_manifest.json: "
     + json.dumps(_final_fingerprint) +
     ". DECIDE both on300 fixed reserved IDs in final_practice_check.json (same IDs as passes5/10), disjoint from500 fit and100 tune for this saved candidate. "
     "This deterministic repeat is an integrity check, NOT additional independent evidence; do not combine sample sizes. "
     "Rejected full-practice refits are not deployed. Final pass25 will repeat hidden validation for deployment confirmation only.\n")
print("Ready:", str(_req24_path))
```
