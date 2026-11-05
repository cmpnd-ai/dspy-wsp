I’ll compare the read outputs with the original features and the local scorer calculation. Then I’ll reverse the correction direction as a negative control: if the direction matters, reversing it should undo the benefit.

```python
# Hypothesis: the causal change is a directional readout repair, with no altered upstream representation.
origH=np.load(work/"fwd_2/head_inputs.npy")
editH=np.load(work/"fwd_21/head_inputs.npy")
editL=np.load(work/"fwd_21/option_logits.npy")
print("Head input max difference",np.nanmax(np.abs(origH-editH)))
print("Scorer replay max error",np.nanmax(np.abs(np.einsum("qod,d->qo",A,wC)-editL)))
reversew=np.asarray(w0)-dc
revscore=measure(np.einsum("qod,d->qo",At,reversew),rt,Mt,yt)[0]
print("Reverse predicted accuracy",np.mean(revscore))
save_edit("head.scorer.3.weight",torch.tensor(reversew).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass 21 feature verification: max head-input difference "+str(float(np.nanmax(np.abs(origH-editH))))+
            "; scorer replay error "+str(float(np.nanmax(np.abs(np.einsum('qod,d->qo',A,wC)-editL)))))+
            ". Next pass 22 reverse direction w0-(wC-w0), same 420 testids, diagnostic negative control.\n")
```
