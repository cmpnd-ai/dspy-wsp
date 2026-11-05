Submitted the validated edit to **`scalar.weight` only**. All backbone weights and other decision-head tensors remain unchanged.

The main failure was the shared scalar readout: it showed near-fixed option preferences across varied records. I replaced its readout direction with a closed-form, regularized correction derived from 600 practice examples—without gradients or backbone changes.

Causal checks supported this localization:
- Increasing correction strength improved a fixed diagnostic panel from **37.0% to 60.5%** before gains saturated.
- Reversing the direction reduced accuracy to **31.5%**.
- State-dependent corrections outperformed fixed per-option offsets by **6.27 percentage points** on independent practice cases.

**Independent results** (gain ± paired standard error):

| Evaluation | Original | Edited | Gain |
|---|---:|---:|---:|
| Untouched practice, 300 cases | 36.20% | 57.47% | **+21.27 ± 1.54 pp** |
| Held-out validation, 300 cases | 34.80% | 57.20% | **+22.40 ± 1.56 pp** |

The final validation run reproduced the earlier result exactly: **198 cases improved and 33 worsened**. The saved edit is that validated candidate, not a diagnostic ablation.