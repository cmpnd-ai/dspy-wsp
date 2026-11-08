I’ll compare the original and repaired responses across the controlled invoice changes. I’ll also separate constant option preferences from state-dependent corrections, so any next edit addresses evidence use rather than merely favoring “discrepancy.”
```python
# Hypothesis: numerical mismatches remain weakly encoded/read, whereas delivery-condition cues should be recoverable by a targeted scorer direction.
m15=json.load(open(W/"fwd_15"/"meta.json"))
in15=np.load(W/"fwd_15"/"head_inputs.npy")
sf15=scorer_features(in15)
print("scorer_features outputs", [tuple(v.shape) for v in sf15] if isinstance(sf15,tuple) else type(sf15))
print("Conditional grid dev:", {str(rel):evaluate_logits(base_linear+add)["acc"] for rel,add in allcond.items()})
print("Target shape",target_mode.shape,"target range",float(target_mode.min()),float(target_mode.max()))
print("Conditional signal decomposition",float(prior_add.square().mean().sqrt()),float(state_add.square().mean().sqrt()))
print("First read rows",m15["rows"][:2])
print("First invoice labels",json.dumps(by_id[inv_fit_ids[0]]["labels"]["questions"])[:2200])
```
