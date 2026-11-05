The coordinate ablations show a distributed effect: accuracy rises from 47.62% with 64 corrections to 58.81% with 256, 67.78% with 512, and 80.63% with all 1,024. I’ll restore C and verify directly that its edit changes the final logits while leaving the scorer’s input features unchanged.

```python
# Hypothesis: C acts entirely through the last linear readout; upstream decision features remain identical.
save_edit("head.scorer.3.weight",candidateC)
with (work/"notes.md").open("a") as f:
    f.write("\nPass 20 top512 .67778, gain .15794 SE .02011. Restored full C. Next pass 21 READ edited on 420 fitids (980 questions), to verify unchanged head features and predicted logits. Only head.scorer.3.weight edited.\n")
req=json.loads((work/"read_fit.json").read_text())
req["model"]="edited"
(work/"verify_fit.json").write_text(json.dumps(req))
print("Read directories",[(p.name,[x.name for x in p.iterdir()][:8]) for p in work.glob("fwd_*")])
```
