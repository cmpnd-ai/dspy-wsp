The half-dose result matches the prediction: only a 2.5-point gain, with most original choices unchanged. I’ll next test a 90% dose to check the transition toward the full repair.

```python
# Hypothesis: progressively removing the old readout bias will progressively release record-dependent decisions.
save_edit("head.scorer.3.weight",torch.from_numpy(w0+.9*dC).reshape(1,-1))
with open(work/"notes.md","a") as f:
    f.write("Pass 11 C half-dose actual .54524 vs .51984, gain .02540 SE .00613. Next 90% C dose on 420 explored test cases.\n")
```
