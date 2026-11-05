Submitted **candidate C**, changing only `head.scorer.3.weight`. The encoder and all other head parameters remain unchanged.

The diagnosis localized a substantial failure to the final readout: the existing head features contained useful decision evidence, but the original scoring direction read it poorly. I replaced that direction using a closed-form ridge solution on centered option features from 840 practice cases—without gradients or backbone changes.

**Measured results**
- **Held-out validation, 300 cases:** 54.33% → **73.00%**, a gain of **18.67 percentage points ± 2.68 pp paired SE**; 134 cases improved and 41 worsened.
- **Practice reserve, 60 cases excluded from fitting:** 52.78% → **63.33%**, gain **10.56 ± 6.16 pp**. This small sample is inconclusive alone.
- **420 fit-included diagnostic cases:** 51.98% → **80.63%**, gain **28.65 ± 2.26 pp**; not a generalization estimate.

Causal tests supported a directional, distributed readout repair: partial-strength and sparse-coordinate edits recovered only part of the gain, reversing the direction slightly hurt accuracy, and option-bias-only correction failed. Head inputs remained exactly unchanged, while the edited logits matched the recomputed scorer.

The final validation rerun reproduced C’s earlier result exactly. The confirmed C tensor is left installed.