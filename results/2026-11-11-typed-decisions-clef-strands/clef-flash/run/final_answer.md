**Submitted the saved `best_2` edit**, left in `work/edited`.

### What changed
Only three head tensors were edited:
- `residual_scorer.0.weight`
- `residual_scorer.0.bias`
- `residual_scorer.3.weight`

The repair repurposes **24 low-contribution scorer neurons**: two provide a shared linear correction through opposite GELU activations; 22 provide gated corrections for agent urgency and customer action/human-review decisions. Directions were computed by closed-form ridge solves on practice data—no gradients or iterative training. The backbone, attention routing, priors, and global scales remain unchanged.

### Evidence for the repair
The failure localized to the final scorer: useful state-sensitive contrasts existed in its field/option inputs but were poorly reflected in option margins. Directly adjusting the existing output weights did not help; exposing these contrasts through the replacement neurons did.

Causal checks showed:
- Graded responses as repair strength changed.
- **Exactly unchanged upstream scorer inputs**, priors, and cosine scores.
- Predicted versus actual edited logits agreeing within **0.00043**.
- On independent-seed probes, one tool error shifted the partial-versus-success margin from **−0.35 to +0.39**; constraint violations also produced substantially stronger urgency responses.

### Measured results

| Evaluation | Original | Edited | Gain ± paired SE |
|---|---:|---:|---:|
| Reserved practice, 300 records | 70.53% | 76.20% | **+5.67 ± 0.99 pp** |
| Hidden validation, 300 records | 70.20% | 77.07% | **+6.87 ± 0.90 pp** |

Validation had **110 improved records and 25 worsened**. Final deployment checks exactly reproduced the earlier results; repeated runs were not counted as additional samples.

The shared two-neuron repair accounts for most of the clearly established gain. The gated extension’s incremental benefit remains uncertain on reserved practice, although the combined edit achieved the best confirmed validation score. Duplicate detection and some customer-review cues remain imperfect.

I expect this edit to help because it improves how existing state evidence affects decisions—not merely confidence—and its overall gain substantially exceeds the paired standard error on both evaluation sets.