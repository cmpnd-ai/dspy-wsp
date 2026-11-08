Gain 4 made violation-sensitive decisions stronger, but it added collateral errors and scored slightly below gain 2 overall. I’ll restore gain 2 and test the combined payment/violation edit on the 450 confirmation cases, which were not used to construct or choose the directions.
```python
# Hypothesis: payment gain4 plus violation gain2 generalizes better than the higher-dose violation edit.
print("Installer variables:", install_linear_cues.__code__.co_varnames)
print("Installer constants:", [str(c)[:650] for c in install_linear_cues.__code__.co_consts if not inspect.iscode(c)])
print("Cue row assignments:", cue_neurons, violation_neurons)
print("Checkpoint files:", sorted(p.name for p in (WORK/"payment4_violation2").iterdir()))
valid_head = arr3["option_mask"].astype(bool)
print("Max valid exported-feature difference:", float(np.max(np.abs(arr11["head_inputs"][valid_head]-arr3["head_inputs"][valid_head]))))
note("\nPass11 payment4+violation4: discovery B .585 vs .557, paired +.0280 SE .01099. Increment over violation2 -.0030 SE .00539. Violated outcome10/10, risk10/10, urgency8/10; collateral errors increase. Prefer violation2. Features compare only at valid options (padding may contain NaNs).")
```
