A probability-linear target performs slightly better internally than log-probabilities and avoids large targets for near-zero gold probabilities. I’ll recompute that correction using all 600 fitting cases, freeze it, and evaluate it on the untouched 300-case practice test split.

```python
# Hypothesis: a probability-linear readout correction generalizes better while repairing the same candidate-collapse mechanism.
Xall=np.concatenate([d[0] for d in designs])
yall=target_residual(sum([d[2] for d in designs],[]),"linear")
Gall=Xall.T@Xall
ea,Ua=np.linalg.eigh(Gall)
dw600=Ua@((Ua.T@(Xall.T@yall))/(ea+.045))
save_edit("scalar.weight",torch.from_numpy((w0+dw600).astype(np.float32))[None,:])
np.save(work/"dw600_linear.npy",dw600)
json.dump({"type":"decide","model":"both","problems":test_ids},open(work/"test_request.json","w"))
with open(work/"notes.md","a") as f:
    f.write("\nWithin-fitting target comparison: linear target 4*p, lambda .03, strength 1 gives .553 internal accuracy versus log .531. Selected using fitting split only.\n"
            "Frozen candidate: refit linear target on 600 fitting cases, lambda .045 (sample-size-scaled), scalar = original + dw600; all other weights untouched.\n"
            "Pass 7: paired decide on untouched 300 practice test cases. No test labels used to choose candidate.\n")
print("Correction norm",np.linalg.norm(dw600),"largest changed coordinates",np.argsort(abs(dw600))[-10:].tolist())
```
