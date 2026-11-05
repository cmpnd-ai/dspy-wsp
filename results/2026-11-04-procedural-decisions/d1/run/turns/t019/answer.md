A direct readout with regularization 0.1 improves the practice replay to 72.1%, especially on sensor evidence. I’ll test that candidate and record a paired request; candidate A remains backed up.

```python
# Hypothesis: less regularization exposes sensor-evidence directions that candidate A still underuses.
candidateB=torch.from_numpy(direct[.1]).reshape(1,-1)
save_edit("head.scorer.3.weight",candidateB)
torch.save(candidateB,work/"candidate_B.pt")
with open(work/"notes.md","a") as f:
    f.write("Candidate B: final scorer readout direct centered gold ridge lambda=.1, fit same 420. Practice replay test .72063 vs baseline .51984 (+.20079 SE .0238); sensors .8429 vs A .7357. Next paired practice test run, B only final readout edited.\n")
```
