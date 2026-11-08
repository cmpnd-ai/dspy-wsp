Kept the validated edit: **384 of 2,048 gains** in `layers.23.post_ffn_norm.weight`. All other tensors, including embeddings and the final norm, remain unchanged.

The repair selectively reweights the last MLP’s contribution to the residual stream. Two independent causal gain probes localized decision-sensitive dimensions; a closed-form ridge solution derived the correction from practice examples. Strength tests showed a dose-response and degradation at excessive strength.

| Evaluation | Original → Edited | Paired gain ± SE |
|---|---:|---:|
| Independent practice holdout, 450 records | 61.91% → 68.44% | **+6.53 ± 1.14 points** |
| Hidden validation, 300 records | 60.67% → 67.93% | **+7.27 ± 1.31 points** |

Overall practice accuracy rose from **61.33% to 68.64%**, though that includes fitting examples.

This is not merely a confidence adjustment: it changed the winning option on 700 of 2,250 holdout questions. Holdout mean Brier improved from **0.1616 to 0.1102**. Some question types regressed, but all four workflows improved overall on the practice holdout. The candidate was locked before holdout and validation testing, and the independent gains support keeping it.