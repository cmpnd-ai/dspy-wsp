Submitted the validated edit to **`scalar.weight` only**. All backbone weights, normalization parameters, biases, and set-attention weights remain unchanged.

The repair changes how the scalar head reads existing hidden-state evidence. I derived an additive direction using closed-form ridge regression on practice option contrasts—no gradient training or global temperature scaling.

Evidence supporting the mechanism:
- An option-prior-only control produced essentially no gain.
- Reversing the correction reversed sensitivity to several decisive state cues.
- A strength sweep favored the submitted correction over both half strength and 1.5× strength.

**Measured results**
- **Held-out validation, 300 examples:** 45.11% → **68.67%**, a gain of **23.56 percentage points ± 3.02 pp paired SE**; 148 examples improved and 51 worsened.
- **All 900 practice examples:** 48.00% → **80.30%**, a gain of **32.30 pp ± 1.49 pp paired SE**. This is in-sample for the final edit.

The final validation reproduced the earlier result exactly. The evidence supports a useful readout repair, not complete reasoning recovery: intent and some counterfactual probes remain weak.