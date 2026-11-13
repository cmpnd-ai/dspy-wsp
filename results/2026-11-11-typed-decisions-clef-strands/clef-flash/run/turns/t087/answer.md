The tensors match the saved best edit exactly. I’ll summarize the new probe margins and record the next run. These are unscored mechanism tests, not new accuracy estimates.

```python
# Hypothesis: tool-error and constraint-violation responses recur on independent seeds, while known weak cues remain weak.
_probe22_lookup = {(r["request"], r["question"]): (i, r["options"]) for i, r in enumerate(_rows22)}
_probe22_models = (_original22, _shared22, _full22)
_probe22_stats = {}
_action_expectations = {
    "routine_information": "answer_directly",
    "human_authority": "escalate_to_human",
    "resolved": "close_no_action",
    "verified_refund": "execute_refund",
    "missing_information": "request_information",
}
for _ri, _pm in enumerate(held_probe_meta):
    _fam, _level = _pm["family"], str(_pm["level"])
    if _fam == "tool_errors":
        _specs = [("outcome", "partial", "success")]
    elif _fam == "constraint_violations":
        _specs = [("urgency", "3", "1")]
    elif _fam == "duplicate_invoice_id":
        _specs = [("duplicate", "true", "false")]
    elif _fam == "benign_security_context":
        _specs = [("credential_compromise", "true", "false"), ("true_positive", "true", "false")]
    else:
        _specs = [("action", _action_expectations[_level], None), ("needs_human", "true", "false")]
    for _q, _pos, _neg in _specs:
        _ix, _opts = _probe22_lookup[(_ri, _q)]
        _pi = _opts.index(_pos)
        _values = []
        for _arr in _probe22_models:
            if _neg is None:
                _others = [j for j in range(len(_opts)) if j != _pi]
                _margin = _arr[_ix, _pi] - _arr[_ix, _others].max()
            else:
                _margin = _arr[_ix, _pi] - _arr[_ix, _opts.index(_neg)]
            _values.append(float(_margin))
        _key = (_fam, _level, _q, _pos + "-" + (_neg or "best_other"))
        _probe22_stats.setdefault(_key, []).append(_values)

_probe22_summary = []
print("family / level / question / margin: original, shared, full (means)")
for _key, _vals in _probe22_stats.items():
    _means = np.mean(_vals, axis=0)
    _probe22_summary.append({"key": list(_key), "n": len(_vals), "means": _means.tolist()})
    print(" / ".join(_key) + ":", np.round(_means, 3).tolist())

(_handoff_work / "held_cue_probe_summary.json").write_text(json.dumps(_probe22_summary, indent=2))
note("\n## Pass 22 independent-seed mechanism summary\n"
     "Best_2 tensor equality verified exactly against targeted_candidate.pt. Original/shared/full margins on40 seeds excluded from best_2 fit+tune: "
     + json.dumps(_probe22_summary) +
     ". These are unscored probes, not independent accuracy estimates. No fitting or weight selection.\n")
_req23_path = _handoff_work / "held_cue_probes_edited.json"
_req23_path.write_text(json.dumps({"type": "read", "model": "edited", "requests": held_probe_requests}))
note("\n## Planned pass 23 — independent-seed causal verification\n"
     "Exact edit remains best_2, targeted_candidate.pt: 24 neurons in residual_scorer.0.weight/.bias and residual_scorer.3.weight; no other edits. "
     "READ edited on same170 requests/850 questions as pass22, held_cue_probes_edited.json. "
     "40 seed IDs are recorded in held_cue_probe_meta.json; no new labels or fitting. Verify scorer input invariance and predicted logit changes. "
     "Final two passes will repeat reserved-practice paired evaluation and validation only to verify final deployment, not select hyperparameters.\n")
print("\nPass 23 request:", str(_req23_path))
```
