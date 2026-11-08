**Submitted:** only `scalar.weight` was changed. The backbone, normalization layers, scalar bias, and set-attention head remain untouched.

### Repair and evidence
The original readout strongly favored fixed answers—every boolean was “true,” agent actions were usually “stop,” and customer categories were usually “billing.” I derived a **closed-form, regularized correction to the scalar readout** from centered option-feature contrasts. This is a directional edit, not global rescaling.

Evidence that it recovers usable state information:
- Backbone features and set-attention corrections remained exactly unchanged.
- Restricting the repair to mean question/option directions lost **7.09 ± 0.83 percentage points** in practice cross-fitting.
- On matched customer-topic probes, correct classifications increased from **6/20 → 14/20 → 20/20** at zero, half, and full edit strength.

### Measured results
Gains below include paired standard errors.

| Evaluation | Original | Edited | Gain |
|---|---:|---:|---:|
| Practice cross-fit, 900 records | 36.42% | 57.16% | **+20.73 ± 0.93 pp** |
| Hidden validation, 300 records | 34.80% | 58.07% | **+23.27 ± 1.52 pp** |

Cross-fitting informed strength selection, so it is not an untouched final test. Before the full-data refit, the earlier 600-record candidate also gained **21.07 ± 1.42 pp** on 300 independent practice records.

The final validation run reproduced the frozen edit’s result exactly: **207 records improved and 30 worsened**. That repeat confirms restoration, not additional independent evidence.

The repair should help because it reads existing semantic evidence more effectively while correcting strong answer biases. It is incomplete: deadline reasoning, exact duplicate-ID matching, and some numeric comparisons remain weak.