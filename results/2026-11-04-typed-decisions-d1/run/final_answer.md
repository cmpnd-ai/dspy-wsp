Submitted the full correction to **`head.scorer.3.weight` only**. All other tensors remain unchanged.

The evidence localizes a repairable failure to the final scorer readout: its existing contextual features support better decisions than the original output weights extract. I derived an additive correction using closed-form ridge regression on practice-only scorer features—no gradients or backbone changes.

**Measured results** (± paired standard error):
- **Independent practice test, initial 450-case fit:** +8.00 ± 1.09 percentage points.
- **Final refit, all 900 practice cases:** +8.93 ± 0.74 points, measured in-sample.
- **Final held-out validation, 300 cases:** **57.53% → 65.60%**, a gain of **8.07 ± 1.26 points**; 142 cases improved and 60 worsened.

Causal checks support a contextual readout repair rather than global sharpening: half-strength and excessive-strength edits performed worse, a prior-only ablation produced essentially no validation gain, and a context-only ablation retained a significant benefit. Recorded head inputs stayed identical, while the changed logits matched the readout calculation within 0.000046.

The full validated edit is left in `edited`. The independent practice and validation gains support expecting improvement on unseen records, though the repair does not fix every question or cue response.