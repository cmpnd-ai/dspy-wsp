Both counterfactual reads are complete. I’ll summarize their cue responses and check that their scorer inputs are unchanged, then prepare a closed-form refit using all 900 practice cases. The already validated 450-fit edit remains safely backed up.

```python
# Hypothesis: the repair increases cue sensitivity; more practice data may stabilize the same readout direction.
cm=json.load(open(work/"fwd_11/meta.json"))["rows"]
cp=[np.load(work/f"fwd_{k}/probabilities.npy") for k in [11,12]]
print("Scorer input difference",np.nanmax(abs(np.load(work/"fwd_11/head_inputs.npy")-np.load(work/"fwd_12/head_inputs.npy"))))
for i,r in enumerate(cm):
    tag=cue_tags[r["request"]]
    if (tag[1]=="violations" and r["question"] in ["risk","outcome"]) or (tag[1]=="days_due" and r["question"]=="urgency"):
        opts=r["options"]
        print(tag,r["question"],"original",dict(zip(opts,np.round(cp[0][i,:len(opts)],2))),
              "edited",dict(zip(opts,np.round(cp[1][i,:len(opts)],2))))
# Closed-form refit at the same regularization per observation: twice the data, twice ridge.
aall=torch.tensor(X[M],dtype=torch.float64); yall=torch.tensor(Y[M],dtype=torch.float64)
delta_all=torch.linalg.solve(aall.T@aall+2*torch.eye(1024,dtype=torch.float64),aall.T@yall).float().numpy()
np.save(work/"readout_delta_all.npy",delta_all)
print("All-data norm",np.linalg.norm(delta_all),"cosine to 450-fit",delta_all@delta/(np.linalg.norm(delta_all)*np.linalg.norm(delta)))
save_edit("head.scorer.3.weight",torch.tensor(w0+delta_all).reshape(1,-1))
json.dump({"type":"decide","model":"both","problems":[x["id"] for x in examples[:450]]},open(work/"all_part0.json","w"))
json.dump({"type":"decide","model":"both","problems":[x["id"] for x in examples[450:]]},open(work/"all_part1.json","w"))
with notes.open("a") as f:
    f.write("\nPass11/12 matched cue reads completed; inputs should be identical because readout only changes. Current trial: same closed-form residual readout refit on all 900 practice cases, ridge2 (same per-observation strength); delta saved readout_delta_all.npy. Next both on first 450 practice cases, then remaining 450, then validation. Best_1 retains independently confirmed 450-fit edit.\n")
```
