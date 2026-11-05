# Weight-repair experiment
No edits yet. Hypothesis to investigate: shared readout misses task-relevant directions.
Deterministic split: 600 fitting cases, 300 untouched practice test cases (seed 814).
Next: original baseline on all 900 practice cases.

Pass 1: original model, all 900 practice records; no edits. Baseline diagnosis only.

Baseline: 0.36422 ± 0.00764 item SE. First trace wrongly stops despite zero violations/irreversible actions.
Pass 2 planned: original read, first 200 fitting IDs, no edits, inspect EOS/head representations.

Failure breakdown: agent action 1.3%, customer category 24.4%, invoice duplicate 9.3%; near-fixed candidate preferences across varied states.
Pass 3: original read, next 200 fitting cases; no edits.

Pass 4 planned: original read, final 200 fitting cases; no edits. All candidate edits will be closed-form scalar projections, not backbone changes.

Closed-form scalar correction from first 400 fitting cases: centered log gold probabilities minus original logits, ridge lambda .03. No gradients. Direction norm 61.72 versus original head norm .551.
Internal 200-case head-only dose response: strength 0/.25/.5/.75/1/1.25 => accuracy .363/.429/.509/.531/.524/.511. Set corrections typical within-choice range .142 logits.
Pass 5: actual scalar.weight = original + .25 * dw400; all other tensors original; paired decide on internal 200 fitting cases.

Pass 5 confirmed causal prediction exactly: .363 -> .429, paired gain .066 ± .01153, 72 improved / 22 worsened.
Pass 6: scalar original + .75*dw400, same internal 200 fitting cases, all other weights unchanged.

Pass 6 dose .75 confirmed: .363 -> .531, gain .168 ± .01899; 115 improved, 34 worsened. Scalar intervention reproduces prediction exactly at two doses.

Within-fitting target comparison: linear target 4*p, lambda .03, strength 1 gives .553 internal accuracy versus log .531. Selected using fitting split only.
Frozen candidate: refit linear target on 600 fitting cases, lambda .045 (sample-size-scaled), scalar = original + dw600; all other weights untouched.
Pass 7: paired decide on untouched 300 practice test cases. No test labels used to choose candidate.

Independent practice test (300): original .36200, edited .574667; gain .212667 ± .015428 paired SE, 199 improved / 34 worsened.
Pass 8: held-out validation, scalar = original + dw600_linear, no other edits. Candidate snapshot candidate_linear600.

VALIDATED BEST: scalar.weight original + dw600_linear only. Validation .348 -> .572, gain .224 ± .015571; 198 improved / 33 worsened. Snapshot best_1.
Remaining experiments are mechanism checks, not validation-driven fitting.
Pass 9: original read first 200 independent practice test cases, no change to saved best scalar edit (read explicitly original).

Pass 10: original read remaining 100 independent practice test cases. Best edit unchanged.

Mechanism check on independent 300 practice cases: fixed per-question-option average corrections versus full state-dependent correction. No new weight selection.

Fixed offsets .512 vs full .574667 on 300 test cases: state-dependent correction adds .062667 accuracy.
Passes 11–12: original and validated edited reads of 7 controlled cue variants (3 trace violation counts, 4 customer categories). No fitting or weight changes.

Passes 11–12 completed controlled cue reads; next: algebraic covariance-direction ablations to characterize the validated scalar correction. Best remains original + dw600_linear.

Pass 13: diagnostic dose_zero; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 14: diagnostic dose_quarter; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 15: diagnostic dose_half; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 16: diagnostic dose_threequarters; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Restored best_1 scalar after restart. Pass 17: full validated scalar on same 40 fitting cases + 7 cue variants as passes 13–16; dose-response endpoint, not selection.

Pass 17: diagnostic dose_125; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 18 (corrected numbering): scalar original + 1.25*dw600; fixed 40-case fitting panel and 7 synthetic requests.

Pass 18: diagnostic dose_150; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 19: diagnostic scalar original + 1.5*dw600, same fixed fitting panel and synthetic cues; best_1 unchanged.

Pass 19: diagnostic cov_ge_1; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 20: diagnostic covariance eigenvalue >=1 projection of dw600; scalar only; fixed fitting panel + synthetic cues.

Pass 20: diagnostic cov_ge_point1; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 21: diagnostic covariance eigenvalue >=0.1 projection of dw600; same fixed panel; not a candidate-selection run.

Pass 21: diagnostic cov_ge_point01; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Pass 22: diagnostic covariance eigenvalue >=0.01 projection of dw600, same fixed panel and cues.

Pass 22: diagnostic reverse_direction; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.

Fixed fitting-panel accuracies: dose 0/.25/.5/.75/1/1.25/1.5 = .370/.425/.505/.605/.585/.580/.565 (see actual summary for high-dose values). Pass 23: negative control scalar original -0.5*dw600, same fixed panel.

Correction: dose 1.25/1.5 accuracies were .570/.575. Covariance cutoff 1/.1/.01 accuracies .515/.560/.580. Reversed direction accuracy 0.315. Restored best_1 exactly; no diagnostic variant retained.
Pass 24: paired decide, restored scalar original+dw600_linear, first 450 fitting cases. Descriptive fit performance only; independent test and validation remain the evidence for selection.

Pass 24 fitting-panel result: .363111 -> .648000, paired gain .284889 ± .012895 (450 cases; descriptive only).
Pass 25: final held-out validation confirmation. Exact best_1 scalar.weight restored; all other weights original. No further selection or fitting.
