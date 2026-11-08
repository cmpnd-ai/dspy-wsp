I’ll summarize the matched-cue results and verify that the logits changed exactly as predicted by the scalar edit. Then I’ll run a half-strength diagnostic to check the dose response before restoring and confirming the frozen final weights.
This analysis checks both successful cue responses and remaining weaknesses. The half-strength run is diagnostic only; it will not change the selected final candidate.
```python
# Hypothesis: the frozen scalar correction changes cue-dependent margins linearly while leaving all backbone and set-attention features fixed.
diag_meta22 = json.loads((work/"fwd_22"/"meta.json").read_text())
diag_meta23 = json.loads((work/"fwd_23"/"meta.json").read_text())
assert diag_meta22["rows"] == diag_meta23["rows"]
diag22 = {p.stem:np.load(p) for p in (work/"fwd_22").glob("*.npy")}
diag23 = {p.stem:np.load(p) for p in (work/"fwd_23").glob("*.npy")}
for name in ["head_inputs","path_ends","set_delta"]:
    a,b = diag22[name],diag23[name]
    finite = np.isfinite(a)&np.isfinite(b)
    print(name,"maximum change",float(np.max(np.abs(a[finite]-b[finite]))))
dx = np.where(np.isfinite(diag22["head_inputs"]),diag22["head_inputs"],0)
predicted_change = np.einsum("qod,d->qo",dx,final_direction)
for i,row in enumerate(diag_meta22["rows"]):
    if row["type"]=="noul":
        predicted_change[i,row["options"].index("false")] = 0
vm = diag22["option_mask"].astype(bool)
actual_change = diag23["option_logits"]-diag22["option_logits"]
print("Max scalar-edit reconstruction error:",float(np.max(np.abs(actual_change[vm]-predicted_change[vm]))))
from collections import defaultdict,Counter
diag_groups = defaultdict(list)
for i,row in enumerate(diag_meta22["rows"]):
    m = diagnostic_manifest[row["request"]]
    diag_groups[(m["kind"],row["question"],str(m["value"]))].append(i)
diagnostic_summary = []
for (kind,qid,value), indices in diag_groups.items():
    row = diag_meta22["rows"][indices[0]]
    opts = row["options"]
    po = diag22["probabilities"][indices,:len(opts)]
    pe = diag23["probabilities"][indices,:len(opts)]
    if row["type"]=="noul":
        j = opts.index("true")
        descriptor = "P(true)"
        a,b = float(po[:,j].mean()),float(pe[:,j].mean())
    elif row["type"]=="score":
        levels = np.array([float(o) for o in opts])
        descriptor = "expected level"
        a,b = float((po@levels).mean()),float((pe@levels).mean())
    else:
        target = (value if kind=="topic" else
                  "harmful" if qid=="outcome" else
                  "stop" if kind=="violations" else
                  "approve" if kind in ["duplicate_id","invoice_amount_ratio"] else
                  "contain" if kind=="authorization_evidence" else
                  "escalate_to_human")
        j = opts.index(target)
        descriptor = "P("+target+")"
        a,b = float(po[:,j].mean()),float(pe[:,j].mean())
    line = f"{kind}/{qid}/{value}: {descriptor} {a:.3f}->{b:.3f}"
    if kind=="topic":
        line += f"; correct top {int((po.argmax(1)==j).sum())}/{len(indices)} -> {int((pe.argmax(1)==j).sum())}/{len(indices)}"
    print(line)
    diagnostic_summary.append(line)
note("\nFinal matched-cue tests forward22/23: identical backbone/head-input/set-delta arrays; scalar reconstruction verified.\n"+
     "\n".join(diagnostic_summary))
whalf_final = load_original("scalar.weight")+.5*(frozen_weight-load_original("scalar.weight"))
torch.save(whalf_final,work/"candidate_weights"/"final_half_diagnostic_scalar.pt")
save_edit("scalar.weight",whalf_final)
half_req = {"type":"read","model":"edited","requests":diagnostic_requests}
(work/"final_cue_half_request.json").write_text(json.dumps(half_req))
note("\nPlanned forward24: edited READ same124 matched requests/268 questions, final_cue_half_request.json. Exact temporary diagnostic edit scalar.weight=w0+.5*(best4-w0)=w0+.4375*d_mix900; all other tensors original. Not a candidate-selection run. Test dose response and restore best4 immediately afterward; final forward25 confirms frozen best4 on hidden validation.")
```
