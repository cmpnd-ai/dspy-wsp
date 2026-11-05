The overshoot behaves as predicted. I’ll now split candidate C’s correction into a mean-option subspace and its record-sensitive complement. Testing each separately will show whether merely changing fixed option preferences explains the gain.

```python
# Hypothesis: neither fixed-option bias removal nor record-sensitive evidence alone is sufficient for candidate C.
biasC=(V@(V.T@dC)).astype(np.float32)
evidenceC=dC-biasC
for name,dd in [("bias-only",biasC),("evidence-only",evidenceC)]:
    st,qt=measure(Lt+At@dd,rt,Mt,yt)
    print(name,st.mean(),qt)
save_edit("head.scorer.3.weight",torch.from_numpy(w0+biasC).reshape(1,-1))
with open(work/"notes.md","a") as f:
    f.write("Pass 14 overshoot 1.1 .69603, confirms nonmonotonic dose-response. Next C bias-only ablation: project delta onto rank-13 mean centered question/option subspace. 420 explored practice cases.\n")
```
