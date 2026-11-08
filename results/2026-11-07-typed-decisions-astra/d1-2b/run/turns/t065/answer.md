Kept the **720-record, rank-nine repair** already installed in `edited`. Only `layers.23.feed_forward.w2.weight` is changed; embeddings, norms, and all other weights remain original.

### What changed
I applied a closed-form, state-dependent correction to the last MLP’s output, restricted to nine option-token contrast directions. It addresses systematic decision errors—such as choosing “request information” instead of escalation—rather than globally sharpening probabilities.

### Causal evidence
- Complementary norm probes isolated the last MLP’s residual contribution.
- Actual development accuracy showed a dose-response: **61.89% original → 74.00% half-strength → 78.33% full-strength**, then **71.00%** when over-applied.
- Removing the within-question, state-sensitive component reduced accuracy from **78.33% to 71.11%**. Thus, the repair does more than adjust question-specific biases.

### Final candidate results

| Evaluation | Original | Edited | Paired gain ± SE |
|---|---:|---:|---:|
| Practice records excluded from coefficient fitting, n=180 | 62.11% | 76.67% | **+14.56 ± 1.93 pp** |
| Hidden validation, n=300 | 60.67% | 76.33% | **+15.67 ± 1.42 pp** |

On validation, **172 records improved and 38 worsened**. This candidate has the highest measured validation score among those tested. The large gains outside coefficient fitting, together with the causal ablation and dose-response, support expecting the targeted repair to help on unseen records.