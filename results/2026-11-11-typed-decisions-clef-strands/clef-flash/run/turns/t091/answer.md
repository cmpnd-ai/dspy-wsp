The probe check passed: scorer inputs, priors, and cosine scores are unchanged, and predicted logits match the tool within 0.00043. A missing variable name interrupted the request preparation. I’ll load the saved split file and finish it without changing weights.

```python
# Hypothesis: the persisted split confirms that the final practice check excludes this candidate's fit and tuning records.
_saved_splits = json.loads((_handoff_work / "practice_splits.json").read_text())
print("Saved split sizes:", {k: len(v) for k, v in _saved_splits.items()})
_final_fit_ids = set(_saved_splits.get("fit", _saved_splits.get("fit_ids", [])))
_final_tune_ids = set(_saved_splits.get("tune", _saved_splits.get("tune_ids", [])))
assert len(_final_fit_ids) == 500 and len(_final_tune_ids) == 100
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
