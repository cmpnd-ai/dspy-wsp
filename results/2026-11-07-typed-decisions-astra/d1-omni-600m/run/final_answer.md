Left the **best validation-confirmed repair** in `work/edited`.

### What changed
Only three scorer tensors:
- `head.scorer.1.weight`
- `head.scorer.1.bias`
- `head.scorer.3.weight`

I repurposed **110 low-contribution scorer neurons** to carry a shared residual direction and 54 question/option-gated residual channels. These were derived by closed-form regularized solves using 448 fit examples. The encoder, embeddings, normalization parameters, and temperatures remain unchanged.

### Diagnosis and causal evidence
The scorer underused evidence already present in its inputs—particularly constraint violations, payment deadlines, and security impact cues.

Targeted interventions matched the predicted scorer-level effects. Dose-response tests showed that stronger edits eventually overcorrected. Removing state-dependent conditional directions while preserving average option preferences reduced accuracy by **3.39 points on development examples** and **2.11 points on reserved examples**, supporting an evidence-reading mechanism rather than merely changed answer priors.

### Measured results
All gains below are percentage points; SE is paired over examples.

| Set | Original | Edited | Gain ± SE |
|---|---:|---:|---:|
| Development, 224 examples | 55.00% | 70.09% | **+15.09 ± 1.71** |
| Reserved practice, 228 examples | 58.42% | 72.46% | **+14.04 ± 1.65** |
| Hidden validation, 300 examples | 57.53% | 71.73% | **+14.20 ± 1.54** |

Final validation reproduced the earlier result exactly: **171 examples improved, 47 worsened**.

The independent gains, positive across all four workflows, support generalization. Remaining limitations include weak discrimination of numerical invoice discrepancy magnitude and sensitivity to irrelevant metadata; the repair does not solve those completely.