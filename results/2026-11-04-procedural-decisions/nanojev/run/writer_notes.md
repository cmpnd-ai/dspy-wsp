# NanoJev weight repair

Initial state: no edits. Inspecting practice records before original-model baseline.

Baseline all 900: accuracy .4800, SE .01263. Per-question: {'current_owner': 0.5333333333333333, 'current_severity': 0.49, 'is_open': 0.5766666666666667, 'intent': 0.22, 'is_urgent': 0.5066666666666667, 'workflow_impact': 0.5133333333333333, 'incident_real': 0.49333333333333335}
No edits. Read pass 2: fixed fitting subset, 140 examples per workflow (420 total, 980 questions), seed 731. Test subset remaining 480 excluded from fitting.

Read order checked: baseline order, not requested ID order. Initial misaligned exploratory calculations discarded, no weights applied.
Closed-form residual fit: within-question centered multiclass contrasts; boolean true vs fixed zero; ridge=1 selected on internal 84-item holdout (.6190 vs .4960 baseline). Refit on 420 fitting examples.
Pass 3: scalar.weight = original + 0.5 * delta (delta norm 13.8385); all other tensors original. Paired test on 450 disjoint practice items.

Pass 3 half-strength, disjoint 450: .47185 -> .52370, gain .05185 ± .00935 paired SE (76 improved,25 worsened).
Pass 4: scalar.weight original + 1.0*delta, same 450 for dose-response; no other edits.

Pass 4 full strength disjoint 450: .47185 -> .66889, gain .19704 ± .02296 paired SE; 207 improved,76 worsened. Saved best_full_ridge1.
Pass 5 hidden validation: same full-strength scalar edit; no other tensors changed.

Pass 5 VALIDATION confirmed full ridge1: .45111 -> .64222, gain .19111 ± .02981 paired SE; 130 improved,55 worsened. Current confirmed best = best_full_ridge1.
Pass 6: scalar.weight original + 1.5*delta, same 450 practice test items; predicted overshoot relative to full strength.

Pass 6 1.5x overshoot: .61704 vs .47185, gain .14519 ± .02819, worse than 1x (.66889). Restored validated 1x.
Passes 7-8: original-model read on remaining 480 practice records in two 240-record batches; all existing features retained. No new edit.

Original reads now cover all 900. Three-fold practice CV (seed926): ridge .1=.69074, .3=.68519,1=.67259,3=.65926; no backbone training. Fitted new additive scalar direction on 900 with ridge .1.
Pass 9 validation: original+delta900_ridge.1, norm 63.1165; all other tensors original. Earlier validated edit backed up.

Pass 9 VALIDATION delta900 ridge.1: .45111 -> .68667, gain .23556 ± .03020;148 improved,51 worsened. Saved best_900_ridge01.
Pass 10 control: scalar original+prior-only direction, fit using means within question/option identity, no state-varying evidence; same 450 practice diagnostic cohort (now part of larger fit).

Pass 10 prior-only control: .47185 -> .47407, gain .00222 ± .02939. Restored validated delta900.
Passes 11-12: 210 matched cue pairs (30 per question), original then edited read. One decisive fact changed per pair; event owner probe uses final assign event. No probe used to fit.

Counterfactual probe summaries (margin shift,both correct,pair accuracy): {11: {'current_owner': [0.75, 0.167, 0.4], 'current_severity': [0.045, 0.0, 0.3], 'is_open': [0.022, 0.0, 0.5], 'intent': [0.024, 0.0, 0.417], 'is_urgent': [0.109, 0.0, 0.5], 'workflow_impact': [0.547, 0.0, 0.5], 'incident_real': [0.057, 0.0, 0.5]}, 12: {'current_owner': [2.979, 0.3, 0.533], 'current_severity': [1.71, 0.233, 0.5], 'is_open': [0.336, 0.133, 0.567], 'intent': [0.042, 0.0, 0.133], 'is_urgent': [0.536, 0.133, 0.567], 'workflow_impact': [2.968, 1.0, 1.0], 'incident_real': [0.371, 0.0, 0.5]}}
Exact head intervention prediction max error 0.000014.

Pass 13: confirmed best scalar=original+delta900 ridge.1; decide edited on all 900 practice for complete baseline comparison. Counterfactual failures remain in intent and one-sensor OOD probes; no claim of complete reasoning repair.

Pass 13 all-practice (in-sample): .48000 -> .80296, gain 0.32296 ± 0.01486.
Pass 14 causal ablation: original scalar + high-variance portion of delta900 only (eigenvalue >=1 in practice contrast covariance). Same 450 diagnostic practice cohort; remove low-variance evidence component. Not a candidate to keep.

Pass 14 high-only ablation on 450: .47185 -> .68370, gain .21185 ± .02259; confirms loss from removing low-variance directions.
Pass 15 validation: same high-only scalar correction (eigenvalue>=1); full validated edit remains backed up.

Pass 15 high-only validation .62778, gain .17667 ± .03058; inferior to full .68667. Rejected; restored best_900_ridge01.
Passes 16-18: full confirmed scalar correction, paired per-workflow diagnostics on all 300 practice items per workflow (in-sample): event, multi-view, sensors. No other edits.

Passes16-18 in-sample workflow gains: event .53333→.81778 (+.28444±.01945), multi-view .41333→.67111 (+.25778±.02119), sensors .49333→.92000 (+.42667±.03329). Current edit remains validated full900 ridge.1.

Pass19: causal reversal scalar=original-delta900; matched 210 cue pairs, same probes as11/12, no fit. Pass20 planned half-strength and21 1.5x on same probes. Full validated edit backed up.
Pass19 reversal probes {'current_owner': [-0.74, 0.0, 0.233], 'current_severity': [-0.81, 0.0, 0.233], 'is_open': [-0.146, 0.0, 0.5], 'intent': [0.003, 0.0, 0.433], 'is_urgent': [-0.159, 0.0, 0.5], 'workflow_impact': [-0.937, 0.0, 0.117], 'incident_real': [-0.129, 0.0, 0.5]}.
Pass20 scalar original+.5*delta900, same420 matched probe requests.
Pass20 half probe results {'current_owner': [0.932, 0.3, 0.483], 'current_severity': [0.439, 0.0, 0.35], 'is_open': [0.09, 0.067, 0.533], 'intent': [0.017, 0.033, 0.35], 'is_urgent': [0.161, 0.067, 0.533], 'workflow_impact': [0.879, 0.9, 0.95], 'incident_real': [0.107, 0.0, 0.5]}.
Pass21 scalar original+1.5*delta900 on same420 probes.
Pass21 1.5x probes {'current_owner': [2.047, 0.3, 0.567], 'current_severity': [1.271, 0.3, 0.517], 'is_open': [0.247, 0.2, 0.6], 'intent': [0.026, 0.0, 0.033], 'is_urgent': [0.375, 0.1, 0.55], 'workflow_impact': [2.089, 0.8, 0.9], 'incident_real': [0.264, 0.0, 0.5]}.
Pass22: scalar original+1.5*delta900, both on test450 (now in-sample). Diagnostic strength sweep only; validation-confirmed full direction still incumbent.

Pass22 1.5x on450: .47185→.78074 gain .30889±.02394. Pass23 half-strength scalar original+.5delta900, same450; diagnostic only.

Pass23 half on450 .47185→.63704 gain .16519±.01458. Pass24 restored full scalar original+delta900, same450 paired comparison. All other tensors original.

Pass24 full on450 .47185→.80667 gain .33481±.02071; exceeds half (.63704) and1.5x(.78074). Current submission scalar.weight only, original+delta900 ridge.1. Pass25 final hidden validation on this exact restored edit; no other tensors modified. Full-data practice gains are in-sample; independent earlier 450 test +.19704±.02296 for initial420-fit edit; hidden validation is generalization evidence for final edit.
