# Weight-edit experiment

Goal: diagnose typed-decision failures and make a targeted, closed-form block-weight repair. No backbone execution outside the forward tool; no gradient fitting.

Initial edit: none.

Planned forward pass 1: DECIDE original, all 900 practice IDs from trainset.jsonl. Current edit: none. Purpose: baseline diagnosis; no examples used for fitting yet.

Pass 1 baseline: 900 records, mean accuracy 0.613333, item SE 0.008248; 798/900 items imperfect. Output: gen_1.jsonl. No edits.

Baseline breakdown:
- agent_trace_observability/action: acc 0.364; gold->pred [(('human_review', 'stop'), 62), (('observe', 'human_review'), 41), (('observe', 'stop'), 12)]
- agent_trace_observability/needs_review: acc 0.618; gold->pred [(('false', 'true'), 86)]
- agent_trace_observability/outcome: acc 0.680; gold->pred [(('partial', 'success'), 24), (('success', 'partial'), 12), (('partial', 'harmful'), 10)]
- agent_trace_observability/risk: acc 0.653; gold->pred [(('0', '1'), 41), (('1', '2'), 10), (('3', '2'), 10)]
- agent_trace_observability/urgency: acc 0.498; gold->pred [(('0', '1'), 45), (('1', '2'), 23), (('3', '2'), 15)]
- customer_service/action: acc 0.333; gold->pred [(('escalate_to_human', 'request_information'), 100), (('answer_directly', 'request_information'), 30), (('escalate_to_human', 'answer_directly'), 13)]
- customer_service/category: acc 0.898; gold->pred [(('account', 'billing'), 9), (('account', 'technical'), 6), (('refund', 'billing'), 4)]
- customer_service/churn_risk: acc 0.520; gold->pred [(('1', '0'), 37), (('2', '1'), 24), (('2', '0'), 20)]
- customer_service/needs_human: acc 0.724; gold->pred [(('false', 'true'), 43), (('true', 'false'), 19)]
- customer_service/urgency: acc 0.538; gold->pred [(('2', '0'), 33), (('1', '0'), 31), (('3', '2'), 29)]
- invoice_processing/discrepancy_severity: acc 0.533; gold->pred [(('3', '2'), 45), (('2', '0'), 26), (('3', '0'), 19)]
- invoice_processing/disposition: acc 0.484; gold->pred [(('hold', 'approve'), 30), (('manual_review', 'hold'), 23), (('manual_review', 'reject'), 21)]
- invoice_processing/duplicate: acc 0.942; gold->pred [(('false', 'true'), 11), (('true', 'false'), 2)]
- invoice_processing/matches_order: acc 0.809; gold->pred [(('false', 'true'), 34), (('true', 'false'), 9)]
- invoice_processing/urgency: acc 0.578; gold->pred [(('2', '3'), 42), (('2', '1'), 22), (('1', '0'), 11)]
- security_incidents/credential_compromise: acc 0.560; gold->pred [(('false', 'true'), 88), (('true', 'false'), 11)]
- security_incidents/disposition: acc 0.604; gold->pred [(('investigate', 'contain'), 39), (('close_benign', 'investigate'), 16), (('monitor', 'investigate'), 14)]
- security_incidents/severity: acc 0.684; gold->pred [(('2', '3'), 23), (('3', '2'), 8), (('1', '2'), 8)]
- security_incidents/true_positive: acc 0.733; gold->pred [(('false', 'true'), 60)]
- security_incidents/urgency: acc 0.511; gold->pred [(('3', '2'), 46), (('1', '2'), 33), (('2', '3'), 16)]

Practice split fixed with RNG 71031, stratified: fit=360, dev=180, test=360. Split saved to practice_split.json. Diagnostic tables use all baseline records, but weight fitting will use fit only; dev for strength selection; test for independent paired evaluation.

Mechanism hypothesis: final-block state/question features are available but their option-token contrasts are misaligned. Largest error: customer-service action, 100/225 gold escalation -> request_information. Errors also include agent review over-triggering, security credential over-triggering, and question-specific score shifts. Will test a low-rank last-MLP output correction, not temperature or protected tensors.

Planned pass 2: READ original, 180 balanced fit_a records (900 questions), IDs in read_fit_a.json, layers 23 and 24. Current edit: none. Will inspect actual head-input arrays before deriving any intervention.

Head conventions: head_inputs is final normed answer-position state; hidden_mean is state-token average, NOT MLP activations. Plan: isolate last residual write via two complementary coordinate gain interventions at layers.23.post_ffn_norm. Untouched dimensions determine final-RMS scale ratio; perturbed dimensions then reveal their normalized last-write contribution. This avoids any backbone simulation.

Planned pass 3: READ original fit_b (180 balanced records, 900 questions), read_fit_b.json, layers 23/24. Current edit: none.

Pass 3 completed: original fit_b read, 900 questions in fwd_3.

Planned pass 4: READ original dev (180 records, 900 questions), read_dev.json. Current edit: none. Fit remains 360 records; dev is for closed-form regularization/strength selection, not coefficient estimation; test=360 reserved for paired confirmation.

Planned pass 5: READ edited fit_a (same 180 IDs as original pass 2). Exact edit: layers.23.post_ffn_norm.weight even coordinates multiplied by 1.5, odd coordinates original; all other tensors original. Diagnostic only, not a proposed final edit. Untouched odd coordinates anchor the final-RMS ratio; even-coordinate differences isolate the corresponding last-write contributions.

Pass 5 causal check: untouched-coordinate RMS error 5e-08; final-scale ratio median 0.98626. Probe changes option logits (mean |delta|=0.4082).

Planned pass 6: READ edited fit_b (180 IDs in probe_even_fit_b.json). Exact edit unchanged: post_ffn_norm.23 even gains x1.5; odd original. No w2 edit.

Pass 6 completed: even-gain probe fit_b, fwd_6.

Planned pass 7: READ edited dev, 180 IDs in probe_even_dev.json. Exact edit remains only layers.23.post_ffn_norm even coordinates x1.5. Development data will evaluate, not fit, the repair.

Pass 7 completed: even-gain dev probe, fwd_7.

Planned pass 8: READ edited fit_a. Exact edit replaced, not compounded: layers.23.post_ffn_norm odd coordinates x1.5; even coordinates original. IDs in probe_odd_fit_a.json. Together with pass 5 this identifies all final-write coordinates.

Pass 8 odd anchor RMS error 4.37e-08. Complementary probes recover final-write C=b/RMS(final residual), and U=C/post_ffn_gain. Any left-factor edit W2'=(I+B A)W2 changes the captured last-MLP write to U'=U+B A U, allowing head-only RMS/readout recomputation. B will be confined to option contrast directions, not arbitrary backbone directions.

Planned pass 9: READ edited fit_b (probe_odd_fit_b.json). Exact edit only post_ffn_norm.23 odd coordinates x1.5, even original.

Pass 9 completed: odd-gain fit_b probe, fwd_9.

Planned pass 10: READ edited dev, 180 IDs in probe_odd_dev.json. Exact edit remains post_ffn_norm.23 odd coordinates x1.5; even original. After this diagnostic pass the norm edit will be dropped. Proposed repair location remains layers.23.feed_forward.w2.weight only.

Diagnostic norm edits removed. Current edit: none. Final-write features recovered for all fit and dev records (anchor errors ~5e-8). Option-aware pre-gain contrast subspace has rank 9. Targets: gold log-probability contrasts at observed calibration temperatures (probability floor .015), minus baseline contrasts; inverted through analytic final-write/final-RMS/readout Jacobian. Coefficients will be solved once per ridge value, in closed form, on fit=360 records only. No autograd, optimization, or backbone simulation.

Head-only development grid (not actual model passes):
ridge=0.1, best=(0.75, {'acc': 0.7622222222222222, 'brier': 0.06575577315582468, 'gain': 0.1433333333333333, 'se': 0.015054216111070285})
ridge=1.0, best=(1.0, {'acc': 0.7833333333333333, 'brier': 0.05678224698566702, 'gain': 0.16444444444444445, 'se': 0.01790765716435891})
ridge=10.0, best=(1.0, {'acc': 0.78, 'brier': 0.05633860832844284, 'gain': 0.16111111111111112, 'se': 0.017411457105612645})
ridge=30.0, best=(1.0, {'acc': 0.7711111111111111, 'brier': 0.05887240892104883, 'gain': 0.15222222222222223, 'se': 0.016703043273513622})
ridge=100.0, best=(1.0, {'acc': 0.7566666666666667, 'brier': 0.06313575083959877, 'gain': 0.1377777777777778, 'se': 0.01699226748513872})
ridge=300.0, best=(0.75, {'acc': 0.7533333333333333, 'brier': 0.07489180677531972, 'gain': 0.13444444444444445, 'se': 0.01407883766183379})
ridge=1000.0, best=(1.0, {'acc': 0.75, 'brier': 0.07431270707914404, 'gain': 0.13111111111111112, 'se': 0.01639989267517396})
ridge=3000.0, best=(1.0, {'acc': 0.7433333333333333, 'brier': 0.08078181619196237, 'gain': 0.12444444444444445, 'se': 0.015611321518117465})
ridge=10000.0, best=(1.0, {'acc': 0.7377777777777778, 'brier': 0.08966817187371456, 'gain': 0.1188888888888889, 'se': 0.015252659787565364})
Preliminary setting ridge=1.0, alpha=1.0. Saved factors to rank9_closed_form.npz. Current edit: none.

Proposed edit: W2' = W2 + alpha * B A W2, B is 2048x9 orthonormal option-contrast basis; A ridge=1 closed-form fit on 360 fit records. Rank <=9. Unit delta Frobenius / original = 2.639012. All other tensors original. Targets depend on state/question features, not a global temperature.

Planned pass 11: READ edited dev=180 records, candidate_dev.json, alpha=0.5. Exact installed tensor W2_original + 0.5*delta_w2. Expected head-only dev metrics: {'acc': 0.74, 'brier': 0.07841502223018788, 'gain': 0.1211111111111111, 'se': 0.012849433391341262}. This begins actual-model dose-response verification.

Pass 11 actual half-strength READ dev: {'acc': 0.74, 'brier': 0.07841350883245468, 'gain': 0.1211111111111111, 'se': 0.012849433391341262}; predicted-vs-actual logit MAE 0.000163743, argmax agreement 1.000000.

Planned pass 12: READ edited dev, same candidate_dev.json and same 180 records. Exact edit W2_original + 1.0*delta_w2 (ridge=1, rank<=9); norms and all other tensors original. Unit-strength predicted metrics: {'acc': 0.7833333333333333, 'brier': 0.05678224698566702, 'gain': 0.16444444444444445, 'se': 0.01790765716435891}.

Pass 12 actual unit-strength dev metrics: {'acc': 0.7833333333333333, 'brier': 0.05678229033946991, 'gain': 0.16444444444444445, 'se': 0.01790765716435891}. Actual/predicted probabilities agree to <0.0001. Contrasts differ by workflow/question: not a temperature or global option bias.

Planned pass 13: READ edited same 180 dev records, candidate_dev.json. Exact edit W2_original + 1.5*delta_w2 (ridge=1, rank<=9). Expected development accuracy 0.710, below unit strength due to overshoot. All other weights original.

Pass 13 actual 1.5-strength dev: {'acc': 0.71, 'brier': 0.0981292724609375, 'gain': 0.09111111111111113, 'se': 0.023809831217902585}. Actual dose-response accuracies: original .618889, alpha=.5 .740000, alpha=1 .783333, alpha=1.5 .710000. Causal optimum is bounded; excessive correction reverses gains.

Candidate snapshot: candidate_rank9_ridge1/ (not yet validation-confirmed).

Planned pass 14: DECIDE both on all 360 reserved practice-test IDs, test_pair.json. Exact current edit: W2_original + B A_ridge1 W2_original, alpha=1, rank<=9; all other tensors original. Test records were excluded from fit and strength/regularization selection.

Pass 14 independent practice-test result, n=360: original .607778, edited .764444; paired gain +.156667, SE .013341; 208 improved, 49 worsened. This is ~11.7 paired SE above zero. Candidate: rank9 ridge1 alpha1 W2 repair.

Planned pass 15: DECIDE both on hidden validation, n=300. Exact current edit: layers.23.feed_forward.w2.weight = W2_original + B A_ridge1 W2_original, alpha1; only this tensor changed. Validation has not yet been queried.

Pass 15 VALIDATION CONFIRMED, n=300: original .606667 -> edited .757333; paired gain +.150667 SE .014504; 171 improved, 43 worsened. First hidden validation query. Confirmed best copied to best_1/. Exact edit: rank9 ridge1 alpha1 W2 only. Independent practice gain +.156667 SE .013341.

Next analysis (no forward yet): compare closed-form penalties on actual ||delta W2|| rather than ||A||, using only fit coefficients and dev selection; also decompose state-dependent versus question-centroid components. Best_1 remains safe.

Planned pass 16: READ edited dev (candidate_dev.json), diagnostic state ablation. Exact edit W2_original+B A_question W2_original; A_question=A_ridge1 P, P orthogonal projector onto 20 fit question-mean write vectors. Thus within-question state-dependent complement is removed. Predicted accuracy .711111 versus .783333 with full edit. All norm/embedding tensors original. Confirmed full edit preserved in best_1/.

Independent closed-form target variants (fit only) considered beta=0,.2,.5 winner emphasis, coefficient or actual-weight norm ridge=1,3,10; dev scores:
(0.0, 'weight', 1.0, {'acc': 0.7877777777777778, 'brier': 0.055134244588467156, 'gain': 0.1688888888888889, 'se': 0.01802283000932246})
(0.0, 'coefficient', 1.0, {'acc': 0.7833333333333333, 'brier': 0.05678224698566702, 'gain': 0.16444444444444445, 'se': 0.01790765716435891})
(0.2, 'weight', 1.0, {'acc': 0.7822222222222223, 'brier': 0.06085824131399264, 'gain': 0.16333333333333336, 'se': 0.01786041470425089})
(0.5, 'weight', 1.0, {'acc': 0.7822222222222223, 'brier': 0.08123646201441041, 'gain': 0.16333333333333333, 'se': 0.017580179008727274})
(0.0, 'coefficient', 3.0, {'acc': 0.7811111111111111, 'brier': 0.05533194167708228, 'gain': 0.1622222222222222, 'se': 0.017742827220807666})
(0.2, 'weight', 3.0, {'acc': 0.7811111111111111, 'brier': 0.06039607620366311, 'gain': 0.16222222222222224, 'se': 0.01746070440613744})

Pass 16 actual question-centroid-only ablation: {'acc': 0.7111111111111111, 'brier': 0.10934744775295258, 'gain': 0.09222222222222223, 'se': 0.015707118337010734}. Full minus ablated gain 0.072222, paired SE 0.016433. This causally demonstrates value from the within-question state-sensitive directions, beyond question biases.

Planned pass 17: DECIDE both on reserved practice test n=360, test_pair.json. Exact current edit: W2_original+B A_weightnorm_ridge1 W2_original, alpha1, fit=360. This candidate was selected on dev (.787778) using actual-weight Frobenius regularization; relative update norm 1.8534 versus 2.641 prior. All other weights original. Confirmed best_1 untouched.


Pass 17 reserved practice test, n=360: actual-weight-ridge1 candidate original .607778 -> edited .764444; paired gain +.156667 SE .013013; 208 improved, 50 worsened. Same accuracy gain as best_1, with smaller relative weight change 1.8534.

Planned pass 18: DECIDE both on hidden validation n=300. Exact current edit: layers.23.feed_forward.w2.weight = W2_original + B A_weightnorm_ridge1 W2_original, alpha1, rank<=9, fit=360. No other tensors changed. best_1 remains preserved.

Pass 18 VALIDATION: actual-weight-ridge1 original .606667 -> edited .754000, paired gain +.147333 SE .014387; 165 improved, 41 worsened. Confirmed but below best_1 (.757333). Saved as best_weightnorm/. best_1 remains the highest validation score.

Original ridge solution reproduced at relative coefficient error 2.85e-10. Expanded fit solved once, ridge1 alpha1, 540 records (fit+dev), 2700 questions; no test coefficients or validation labels used. Saved candidate_expanded/ and candidate_expanded_factors.npz.

Planned pass 19: DECIDE both on the 360 reserved practice-test IDs (test_pair.json). Exact edit: W2_original+B A_expanded540 W2_original, rank<=9; all other tensors original. Coefficients use 540 fit+dev records, fixed original target, ridge1, alpha1.

Pass 19 expanded540 reserved practice test n=360: original .607778 -> edited .771667; paired gain +.163889 SE .013709; 212 improved, 50 worsened. Expanded minus first candidate +0.007222, paired SE 0.005954 (not enough to establish superiority, but both beat original).

Planned pass 20: DECIDE both hidden validation n=300. Exact current edit W2_original+B A_expanded540 W2_original, rank<=9, ridge1, alpha1, only layers.23.feed_forward.w2.weight changed. Final candidate coefficients use practice fit+dev only; no validation labels accessible.

Pass 20 VALIDATION CONFIRMED expanded540: original .606667 -> edited .758000; paired gain +.151333 SE .014527; 170 improved, 42 worsened. Highest validation score so far, though difference from best_1 is negligible. Saved best_expanded540/.

Final experiment predeclared: add 180 randomly selected, workflow-balanced records from the existing 360 practice-test records to the closed-form fit. Keep basis, log-probability target, ridge1 and alpha1 fixed, with no strength grid. Three READ passes recover their final-write vectors; one DECIDE assesses the other 180 records; one final hidden-validation pass. Coefficients then use 720 practice records only. Earlier 540 candidate remains safe.

Planned pass 21: READ original 180 additional records listed in final_expansion_split.json added_fit, using read_expansion180.json; 900 questions. Current installed edit is expanded540 W2 only, but model=original ignores it. No labels from remaining_check or validation enter the new fit.

Pass 21 completed: original readout for expansion180 (900 questions), fwd_21.

Planned pass 22: READ edited expansion180, probe_expansion180.json. Exact edit: layers.23.post_ffn_norm.weight even coordinates x1.5, odd original. W2 repair removed. All other weights original. Diagnostic only; confirmed best_expanded540 preserved.

Pass 22 expansion even-coordinate probe completed. Untouched-coordinate anchor RMS error 5.01e-08; final-scale ratio median 0.986246.

Planned pass 23: READ edited expansion180, same probe_expansion180.json. Exact edit replaced with layers.23.post_ffn_norm.weight odd coordinates x1.5, even original; W2 and all other tensors original. Diagnostic only; norm edit will be removed after this read.

Pass 23 odd-coordinate probe completed, anchor RMS error 4.38e-08. Diagnostic norm edit removed. Recovered write features for 180 extra practice records. Final closed-form fit uses 720 records / 3600 questions; original basis, target, ridge1, alpha1 unchanged. Candidate saved in candidate_final720/ with factors.

Planned pass 24: DECIDE both on remaining 180 coefficient-excluded practice records in final_check_pair.json. Exact edit: layers.23.feed_forward.w2.weight = W2_original + B A_final720 W2_original, rank<=9, ridge1, alpha1. All other tensors original. These records were previously evaluated for other candidates but never used for coefficient estimation; sample-size extension was predeclared.

Pass 24 final720 coefficient-excluded practice check n=180: original .621111 -> edited .766667; paired gain +.145556 SE .019288; 99 improved, 29 worsened. Final720 minus expanded540 on the same records +-0.003333 SE 0.007953. No further fitting or strength selection.

Planned pass 25 (final forward): DECIDE both hidden validation n=300. Exact current edit W2_original+B A_final720 W2_original, rank<=9, ridge1 alpha1, only layers.23.feed_forward.w2.weight changed. After the result, retain the highest-scoring validation-confirmed candidate; expanded540 remains saved in best_expanded540/ and first candidate in best_1/. No more candidate tuning.
