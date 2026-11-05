Candidate C reaches 73.0% on validation, a gain of 18.7 percentage points with a 2.7-point paired SE. I’ll preserve it as the current best edit, then use causal ablations to determine what parts of the readout repair are necessary.

```python
# Hypothesis: candidate C's gain depends on a coordinated readout direction rather than sharpening logits.
shutil.copytree(work/"edited",work/"best_C",dirs_exist_ok=True)
with open(work/"notes.md","a") as f:
    f.write("Pass 10 C validation .73000 vs .54333, gain .18667 SE .02685; 134 improved/41 worsened. Best confirmed edit C saved best_C/candidate_C.pt. Further passes are causal diagnostics, not validation tuning.\n")
dC=wC-w0
# Prepare exact head-replay predictions for three intervention doses.
for dose in [.5,.9,1.,1.1]:
    st,qt=measure(Lt+At@(dose*dC),rt,Mt,yt)
    print("dose",dose,"score",st.mean(),qt)
save_edit("head.scorer.3.weight",torch.from_numpy(w0+.5*dC).reshape(1,-1))
with open(work/"notes.md","a") as f: f.write("Next paired test: C half-dose, w=w0+.5*(wC-w0), 420 explored practice cases. Best C safely saved.\n")
```
