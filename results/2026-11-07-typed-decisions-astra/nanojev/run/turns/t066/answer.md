I’ll summarize the half-strength response, restore the frozen best edit, and use the final pass to verify its validation score after restoration. This is a reproducibility check, not a new independent estimate.
```python
# Hypothesis: successful cue responses follow the scalar intervention's strength, and restoring best4 reproduces its confirmed validation result.
diag24 = {p.stem:np.load(p) for p in (work/"fwd_24").glob("*.npy")}
valid_logits = vm
midpoint = .5*(diag22["option_logits"]+diag23["option_logits"])
mid_error = float(np.max(np.abs(diag24["option_logits"][valid_logits]-midpoint[valid_logits])))
print("Half-strength logit midpoint max error:",mid_error)
dose_lines = []
for (kind,qid,value), indices in diag_groups.items():
    if kind not in ["topic","violations","cancellation_intent"] or (kind=="violations" and qid!="needs_review") or (kind=="cancellation_intent" and qid!="churn_risk"):
        continue
    row = diag_meta22["rows"][indices[0]]
    opts = row["options"]
    arrays = [d["probabilities"][indices,:len(opts)] for d in [diag22,diag24,diag23]]
    if kind=="topic":
        j = opts.index(value)
        values = [float(p[:,j].mean()) for p in arrays]
        counts_top = [int((p.argmax(1)==j).sum()) for p in arrays]
        line = f"{kind}/{value}: target P {values[0]:.3f}->{values[1]:.3f}->{values[2]:.3f}; top counts {counts_top}/{len(indices)}"
    elif row["type"]=="noul":
        j = opts.index("true")
        values = [float(p[:,j].mean()) for p in arrays]
        line = f"{kind}/{value}: P(true) {values[0]:.3f}->{values[1]:.3f}->{values[2]:.3f}"
    else:
        levels = np.array([float(o) for o in opts])
        values = [float((p@levels).mean()) for p in arrays]
        line = f"{kind}/{value}: expected level {values[0]:.3f}->{values[1]:.3f}->{values[2]:.3f}"
    print(line)
    dose_lines.append(line)
save_edit("scalar.weight",frozen_weight)
assert edited()==["scalar.weight"]
assert torch.equal(load("scalar.weight"),torch.load(work/"best_4_weights.pt",weights_only=True)["scalar.weight"])
assert load("scalar.weight").dtype==torch.float32
final_digest = hashlib.sha256(load("scalar.weight").numpy().tobytes()).hexdigest()
print("Restored final scalar SHA256:",final_digest)
print("Only edited tensor:",edited())
note("\nForward24 half-strength cue diagnostic: maximum midpoint-logit error "+str(mid_error)+
     ". Strength0/half/full responses:\n"+"\n".join(dose_lines)+
     "\nLimitations: exact duplicate-ID cue responded in the wrong direction; deadline sensitivity and numeric monotonicity remain weak; some security/needs_human responses remain overpositive. Do not claim these mechanisms are fixed.")
note("\nPlanned final forward25: paired hidden VALIDATION300 after restoring exact frozen best4 scalar.weight=w0+.875*d_mix900, candidate_weights/mix900_0875_scalar.pt. All other tensors original. SHA256 "+final_digest+
     ". Expected identical to forward19: .348000 -> .580667, gain .232667 SE .015184. Repeat is restoration/reproducibility check, not new independent evidence. Final submission will leave only this scalar.weight edit.")
```
