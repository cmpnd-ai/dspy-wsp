Neither half reproduces the full correction’s gain. I’ll restore the validated full edit and read the matched-cue requests once more to verify its logits against the reconstructed scorer.

```python
# Hypothesis: the final full correction's effects are completely explained by the changed terminal readout.
save_edit("head.scorer.3.weight",load_original("head.scorer.3.weight")+torch.tensor(delta_all,dtype=torch.float32).reshape(1,-1))
with (work/"notes.md").open("a") as f:
    f.write("\nPass23 complementary512 +.00178±.00209; both halves alone ineffective versus full +.09289 diagnostic450. Restored final full all900 ridge2 delta, head.scorer.3.weight ONLY. Pass24 planned edited READ on 15 matched cue requests via cues_edited.json; pass25 final validation confirmation. No further candidate selection.\n")
print(json.load(open(work/"cues_edited.json"))["model"])
print("Scorer feature callable:", features)
print("Cached arrays:",[(k,v.shape) for k,v in list(globals().items()) if isinstance(v,np.ndarray) and (k in ["X","Y","Z","H","w","w0","delta_all"] or v.shape[-1:]==(1024,))][:25])
```
