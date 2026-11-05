The full correction reaches 80.6% on the explored cases. I’ll test a slight overshoot next; the replay predicts that excessive correction will reverse some useful severity preferences, showing this is not a generic sharpening effect.

```python
# Hypothesis: overshooting the repaired direction harms specific choices, unlike a global temperature change.
save_edit("head.scorer.3.weight",torch.from_numpy(w0+1.1*dC).reshape(1,-1))
with open(work/"notes.md","a") as f:
    f.write("Pass 13 full C explored-practice .80635, +.28651 SE .02262 (fit-included). Dose sequence .5=.54524, .9=.71270, 1=.80635. Next overshoot 1.1 same cases; predicted .69603.\n")
```
