The correction generalizes in the head replay: practice accuracy rises from 52.0% to 69.4%, a gain of 17.4 percentage points with a 2.2-point paired SE. Half-strength barely changes decisions; excessive strength reverses useful preferences. I’ll save the full-strength correction and verify it through the actual program.

```python
# Hypothesis: the replayed gain is caused solely by the final scorer readout and will reproduce in full program calls.
candidate=sw["head.scorer.3.weight"]+torch.from_numpy(deltas[1.]).reshape(1,-1)
save_edit("head.scorer.3.weight",candidate)
torch.save(candidate,work/"candidate_readout.pt")
with open(work/"notes.md","a") as f:
    f.write("\nCandidate A: only head.scorer.3.weight = original + ridge residual delta, lambda=1, strength=1; fitted 420 cases. Untouched practice replay 420: .51984 -> .69365, gain .17381 SE .0221. Half strength .52778; 1.5 strength .38016. Next paired actual program on 420 practice test IDs. All other tensors original.\n")
json.dump({"type":"decide","model":"both","problems":testids},open(work/"paired_test.json","w"))
```
