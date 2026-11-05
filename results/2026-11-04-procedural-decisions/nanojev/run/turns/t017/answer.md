Validation confirms the gain: 45.11% → 64.22%, or +19.11 points (paired SE 2.98). I’ll keep that checkpoint. To complete the causal dose-response test, I’ll test 1.5× strength; the fitting analysis predicts that overshooting should lose accuracy by overcorrecting existing preferences.

```python
# Hypothesis: over-amplifying the same direction reduces accuracy, demonstrating a specific evidence correction rather than global sharpening.
with open(notes,"a") as f:
    f.write("\nPass 5 VALIDATION confirmed full ridge1: .45111 -> .64222, gain .19111 ± .02981 paired SE; 130 improved,55 worsened. Current confirmed best = best_full_ridge1."
            "\nPass 6: scalar.weight original + 1.5*delta, same 450 practice test items; predicted overshoot relative to full strength.\n")
save_edit("scalar.weight",torch.from_numpy((W0+1.5*delta).astype(np.float32)[None]))
```
