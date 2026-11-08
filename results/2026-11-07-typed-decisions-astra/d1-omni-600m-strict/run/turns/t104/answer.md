Left the **best validated repair (`best_2`)** in `work/edited`.

### What changed
Modified **12 scorer neurons** across three tensors:
- `head.scorer.1.weight`
- `head.scorer.1.bias`
- `head.scorer.3.weight`

The edits strengthen readouts for **invoice deadlines, explicit agent constraint violations, and security asset criticality/privileges**. Security readouts include a question-specific gate. All encoder weights, normalizations, temperatures and other tensors remain original.

### Evidence for the mechanism
Matched counterfactuals showed that the head represented these cues, but the scorer often still preferred routine urgency, partial outcomes or medium severity.

Actual interventions at multiple strengths produced the predicted selective changes. For example, explicit violations increased the harmful-versus-partial margin while clean traces barely changed. The security gate reduced off-target logit-change RMS from **0.412 to 0.00003**, preserving its severity effect and leaving all 950 unrelated discovery-question rankings unchanged. Exported head inputs remained **bit-identical**, localizing the repair to the scorer—not the encoder.

### Measured results
Gains and standard errors are in percentage points; SEs are paired over cases.

| Evaluation | Original | Edited | Gain ± SE |
|---|---:|---:|---:|
| Discovery, 400 cases—in-sample | 55.75% | 60.35% | **+4.60 ± 0.72** |
| Independent practice, 500 cases | 57.20% | 59.76% | **+2.56 ± 0.71** |
| Hidden validation, 300 cases | 57.53% | 59.80% | **+2.27 ± 0.85** |

The independent-practice result includes the small 50-case reserve, which regressed **−2.00 ± 2.23 points**; improvement was not uniform. An additional violation-gating experiment also failed to improve accuracy and was discarded.

I expect the retained repair to help because it corrects identifiable cue-reading failures, has causal dose-response evidence, and improves both independent practice and hidden validation—not merely development examples.