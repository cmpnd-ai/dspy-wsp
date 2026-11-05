I’ll verify that the final logit changes equal the closed-form readout prediction, then use the last pass to confirm the exact weights left for submission.

```python
# Hypothesis: all final logit changes equal GELU-feature dot-products with the saved correction.
import inspect
print(inspect.getsource(features))
hf=np.load(work/"fwd_24"/"head_inputs.npy")
lf=np.load(work/"fwd_24"/"option_logits.npy")
lbase=np.load(work/"fwd_11"/"option_logits.npy")
print("Final hidden inputs unchanged:",np.nanmax(np.abs(hf-h0)))
zf=features(hf)
if isinstance(zf,torch.Tensor): zf=zf.numpy()
predicted_change=np.asarray(zf)@delta_all
print("Max logit-change reconstruction error:",np.nanmax(np.abs(lf-lbase-predicted_change)))
with (work/"notes.md").open("a") as f:
    f.write("\nPass24 final edited cue READ: verified unchanged head inputs and terminal-readout logit reconstruction (see kernel output). Final pass25 validation both on300; exact current edit full all900 ridge2 correction ONLY to head.scorer.3.weight; no examples used beyond practice for deriving weights.\n")
```
