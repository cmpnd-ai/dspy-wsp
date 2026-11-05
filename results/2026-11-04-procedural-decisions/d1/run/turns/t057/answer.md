The verification block had a syntax error and did not run. I’ll rerun it with simpler logging.

```python
# Hypothesis: upstream features remain fixed, and reversing the readout correction removes its benefit.
origH=np.load(work/"fwd_2/head_inputs.npy")
editH=np.load(work/"fwd_21/head_inputs.npy")
editL=np.load(work/"fwd_21/option_logits.npy")
hdiff=float(np.nanmax(np.abs(origH-editH)))
lerr=float(np.nanmax(np.abs(np.einsum("qod,d->qo",A,wC)-editL)))
print("Head input max difference",hdiff,"Scorer replay max error",lerr)
reversew=np.asarray(w0)-dc
print("Reverse predicted accuracy",np.mean(measure(np.einsum("qod,d->qo",At,reversew),rt,Mt,yt)[0]))
save_edit("head.scorer.3.weight",torch.tensor(reversew).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write(f"\nPass 21 feature verification: max head-input difference {hdiff}; scorer replay error {lerr}. Next pass 22 reverse direction w0-(wC-w0), same 420 testids, diagnostic negative control.\n")
```
