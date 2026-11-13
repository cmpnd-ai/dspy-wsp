# Weight-repair experiment

Baseline: no edits. Plan: diagnose option/question failures, localize them in the pointer head, derive only closed-form targeted edits, test strengths, then confirm on untouched practice and validation.

## Before pass 1
Exact edit: none. Original baseline on all 900 practice IDs, listed in baseline_request.json. No examples have yet been used to fit or select edits.

Pass 1: original accuracy 0.579111, SE over 900 items 0.008414; 807 items below perfect. Records: gen_1.jsonl. No edits.

Fixed stratified practice split (seed 2610): fit=540 (135/workflow), dev=180, test=180. IDs in splits.json. Test labels/outputs will not be used for fitting or strength selection. All baseline aggregate groups inspected, but only fit individual failures inspected.

Baseline per question:
- agent_trace_observability/action: 0.4489
- agent_trace_observability/needs_review: 0.6222
- agent_trace_observability/outcome: 0.4889
- agent_trace_observability/risk: 0.4978
- agent_trace_observability/urgency: 0.5600
- customer_service/action: 0.3556
- customer_service/category: 0.8756
- customer_service/churn_risk: 0.5600
- customer_service/needs_human: 0.6400
- customer_service/urgency: 0.5822
- invoice_processing/discrepancy_severity: 0.4756
- invoice_processing/disposition: 0.4000
- invoice_processing/duplicate: 0.9200
- invoice_processing/matches_order: 0.6933
- invoice_processing/urgency: 0.2578
- security_incidents/credential_compromise: 0.6400
- security_incidents/disposition: 0.7333
- security_incidents/severity: 0.6622
- security_incidents/true_positive: 0.7289
- security_incidents/urgency: 0.4400

## Before pass 2
Exact edit: none. READ original fit_a, 180 fit items (45/workflow), 900 questions; IDs and request in read_fit_a.json. Hypothesis: final states contain workflow-specific evidence that the shared pointer readout underweights. Planned reads fit_b, fit_c, dev, test are disjoint; test representations will be held unused until an edit is fixed.

## Before pass 3
Exact edit: none. READ original fit_b, next disjoint 180 fit items, 900 questions. Exact IDs in read_fit_b.json. Pass 2 outputs in fwd_2; inspect array metadata before deriving edits.

Head LayerNorm scale is all ones, bias all zeros. No head edits. Exact cached head reproduction max error 3.09944e-06.

## Before pass 4
Exact edit: none. READ original fit_c, final 180 fitting items, 900 questions. Exact IDs in read_fit_c.json. All 540 fitting cases will then have head-input coverage.

Localization: after head LN, top ten dimensions carry 89.2% of decision and 90.0% of option energy. Mean-query norm 9.8753, query residual RMS norm 2.9504; replacing query by mean changes 4/2700 fit predictions. Candidate: rank-one addition to k.weight along mean-query direction, reading option-state contrasts. No weights changed yet.

## Before pass 5
Exact edit: none. READ original development 180 items/900 questions; IDs in read_dev.json. Development is not used to derive candidate directions, only to choose regularization and strength.

Closed-form experiment: X = within-question centered normalized option activations * (query dot mean-query unit direction)/(sqrt(768)*temperature). Targets are centered log(gold probability, floor .01) minus original relative logits. One eigendecomposition of X'X and closed-form ridge solutions, no gradients or iterative fitting. Fit=540 only.

Rank-one key development grid:
- lambda=0.001, strength=0.5: fit=0.5293, dev=0.4256, paired gain=+0.0189 SE=0.0136, Brier=0.2445
- lambda=0.001, strength=1.0: fit=0.6074, dev=0.4433, paired gain=+0.0367 SE=0.0196, Brier=0.2370
- lambda=0.001, strength=1.5: fit=0.5859, dev=0.4389, paired gain=+0.0322 SE=0.0247, Brier=0.2764
- lambda=0.01, strength=0.5: fit=0.5037, dev=0.4444, paired gain=+0.0378 SE=0.0128, Brier=0.2413
- lambda=0.01, strength=1.0: fit=0.5559, dev=0.4611, paired gain=+0.0544 SE=0.0201, Brier=0.2158
- lambda=0.01, strength=1.5: fit=0.5326, dev=0.4422, paired gain=+0.0356 SE=0.0249, Brier=0.2402
- lambda=0.1, strength=0.5: fit=0.4704, dev=0.4389, paired gain=+0.0322 SE=0.0120, Brier=0.2419
- lambda=0.1, strength=1.0: fit=0.5078, dev=0.4678, paired gain=+0.0611 SE=0.0192, Brier=0.2086
- lambda=0.1, strength=1.5: fit=0.4856, dev=0.4544, paired gain=+0.0478 SE=0.0260, Brier=0.2261
- lambda=1.0, strength=0.5: fit=0.4567, dev=0.4367, paired gain=+0.0300 SE=0.0107, Brier=0.2494
- lambda=1.0, strength=1.0: fit=0.4700, dev=0.4456, paired gain=+0.0389 SE=0.0185, Brier=0.2127
- lambda=1.0, strength=1.5: fit=0.4485, dev=0.4500, paired gain=+0.0433 SE=0.0243, Brier=0.2236
- lambda=10.0, strength=0.5: fit=0.4367, dev=0.4300, paired gain=+0.0233 SE=0.0085, Brier=0.2655
- lambda=10.0, strength=1.0: fit=0.4493, dev=0.4289, paired gain=+0.0222 SE=0.0137, Brier=0.2272
- lambda=10.0, strength=1.5: fit=0.4278, dev=0.4256, paired gain=+0.0189 SE=0.0215, Brier=0.2177
- lambda=100.0, strength=0.5: fit=0.4237, dev=0.4111, paired gain=+0.0044 SE=0.0035, Brier=0.2842
- lambda=100.0, strength=1.0: fit=0.4322, dev=0.4189, paired gain=+0.0122 SE=0.0061, Brier=0.2603
- lambda=100.0, strength=1.5: fit=0.4370, dev=0.4289, paired gain=+0.0222 SE=0.0083, Brier=0.2396

IMPORTANT alignment correction: READ-from preserves baseline-file order, not supplied randomized ID order. Initial ridge grid from Python run 8 used misaligned labels and is INVALID/discarded; no edits were saved. Matched every request to baseline completion probabilities (max error <1e-5); verified one-to-one mapping and corrected labels. All subsequent analyses use read_ids matched from actual probabilities.

Correctly aligned rank-one key grid:
- lambda=0.001, strength=0.5: fit=0.7385, dev=0.7233, gain=+0.1456 SE=0.0164, Brier=0.0909
- lambda=0.001, strength=1.0: fit=0.7778, dev=0.7478, gain=+0.1700 SE=0.0213, Brier=0.0733
- lambda=0.001, strength=1.5: fit=0.7622, dev=0.7100, gain=+0.1322 SE=0.0247, Brier=0.0960
- lambda=0.01, strength=0.5: fit=0.7015, dev=0.7033, gain=+0.1256 SE=0.0161, Brier=0.1014
- lambda=0.01, strength=1.0: fit=0.7430, dev=0.7344, gain=+0.1567 SE=0.0198, Brier=0.0804
- lambda=0.01, strength=1.5: fit=0.7352, dev=0.7222, gain=+0.1444 SE=0.0220, Brier=0.0939
- lambda=0.1, strength=0.5: fit=0.6611, dev=0.6589, gain=+0.0811 SE=0.0129, Brier=0.1205
- lambda=0.1, strength=1.0: fit=0.6893, dev=0.6956, gain=+0.1178 SE=0.0171, Brier=0.1006
- lambda=0.1, strength=1.5: fit=0.6904, dev=0.6778, gain=+0.1000 SE=0.0199, Brier=0.1084
- lambda=1.0, strength=0.5: fit=0.6237, dev=0.6222, gain=+0.0444 SE=0.0106, Brier=0.1426
- lambda=1.0, strength=1.0: fit=0.6411, dev=0.6444, gain=+0.0667 SE=0.0137, Brier=0.1268
- lambda=1.0, strength=1.5: fit=0.6404, dev=0.6322, gain=+0.0544 SE=0.0170, Brier=0.1263
- lambda=10.0, strength=0.5: fit=0.5985, dev=0.5922, gain=+0.0144 SE=0.0082, Brier=0.1633
- lambda=10.0, strength=1.0: fit=0.6141, dev=0.6133, gain=+0.0356 SE=0.0114, Brier=0.1525
- lambda=10.0, strength=1.5: fit=0.6241, dev=0.6156, gain=+0.0378 SE=0.0130, Brier=0.1466
- lambda=100.0, strength=0.5: fit=0.5796, dev=0.5789, gain=+0.0011 SE=0.0033, Brier=0.1761
- lambda=100.0, strength=1.0: fit=0.5826, dev=0.5822, gain=+0.0044 SE=0.0050, Brier=0.1730
- lambda=100.0, strength=1.5: fit=0.5852, dev=0.5878, gain=+0.0100 SE=0.0065, Brier=0.1702

Saved k.weight = original + 0.5 * outer(u,v), rank-one key correction; u,v in candidate_rank1/derivation.pt, lambda=0.001. All other tensors original.

Causal controls on dev: original (0.5777778029441833, 0.0, 0.0, 0.17925088107585907); static fit-schema offsets (0.6277777552604675, 0.05000000074505806, 0.018502269115469715, 0.1290670782327652); state-dependent rank1 (0.7477777600288391, 0.17000000178813934, 0.021343747313461822, 0.07331855595111847); within-schema shuffled state corrections (0.54666668176651, -0.031111113727092743, 0.021880677051170153, 0.20072485506534576).

## Before pass 6
Exact edit: k.weight = K0 + 0.5*outer(u,v), lambda .001, fit=540 only. Paired DECIDE on 180 dev IDs in paired_dev.json; first actual dose-response test.

Pass 6 actual paired dev, strength .5: original .577778 -> edited .723333, gain .145556, paired SE .016363; 98 improved, 20 worsened. Matches cached-head prediction.

Saved k.weight = original + 1 * outer(u,v), rank-one key correction; u,v in candidate_rank1/derivation.pt, lambda=0.001. All other tensors original.

## Before pass 7
Exact edit: k.weight = K0 + 1.0*outer(u,v), lambda .001, all other tensors original. Same 180 dev IDs in paired_dev.json. Direction unchanged; full-strength causal test.

Pass 7 actual paired dev, strength 1.0: .577778 -> .747778, gain .170000, paired SE .021344; 103 improved, 32 worsened. Matches cached-head prediction.

Saved k.weight = original + 1.5 * outer(u,v), rank-one key correction; u,v in candidate_rank1/derivation.pt, lambda=0.001. All other tensors original.

## Before pass 8
Exact edit: k.weight = K0 + 1.5*outer(u,v), lambda .001; all other tensors original. Same 180 development cases in paired_dev.json. Overshoot dose-response test; full strength remains provisional best, not yet validation-confirmed.

Pass 8 actual paired dev, strength 1.5: .577778 -> .710000, gain .132222, SE .024723. Dose response strengths 0/.5/1/1.5 gives accuracy .5778/.7233/.7478/.7100; overshoot confirmed. Select strength 1.0 based only on development.

Saved k.weight = original + 1 * outer(u,v), rank-one key correction; u,v in candidate_rank1/derivation.pt, lambda=0.001. All other tensors original.

## Before pass 9
Exact edit: k.weight = K0 + outer(u,v), lambda .001, rank one; no other edits. Paired DECIDE on reserved 180 practice-test IDs (45/workflow) in paired_test.json. These examples were not used to derive or select the edit; no test read has been performed.

Pass 9 reserved practice-test, strength 1.0: original .591111 -> edited .726667, gain .135556, paired SE .021871; 98 improved, 37 worsened. Candidate fixed before test; gain exceeds six paired SE.

## Before pass 10
Exact edit: k.weight = K0 + outer(u,v), lambda .001; all other weights original. Hidden VALIDATION both on all 300 cases; no validation examples or per-item records are visible.

Pass 10 hidden validation: original .582000 -> edited .728667, gain .146667, paired SE .015527; 163 improved, 43 worsened. Rank-one key edit is validation-confirmed.

Current best: best_1/edited, validated rank-one k.weight edit. Counterfactual plan: 10 paired examples each for delivery accepted/exceptions, due in 20/1 days, 0/2 constraint violations, source allowlist false/true. All derive from fit items; no gold labels invented.

## Before pass 11
Exact current edit: validated k.weight = K0 + outer(u,v), lambda .001; READ model=original ignores edit, on 80 synthetic single-cue counterfactual requests (400 questions), exact contents counterfactual_original.json. Pairs and expected margin signs in counterfactual_pairs.json.

Counterfactual predicted sensitivity (positive means expected direction):
- delivery_exceptions: mean original/edited=[0.07690322356386516, 1.4483779125919651], positive pairs=[10, 10]/10
- payment_deadline: mean original/edited=[1.155067678571974, 4.757725759743101], positive pairs=[10, 10]/10
- constraint_violation: mean original/edited=[0.727781351312053, 3.1572797193992708], positive pairs=[10, 10]/10
- source_allowlisted: mean original/edited=[0.11406780958288894, 0.3777560378666618], positive pairs=[10, 9]/10

## Before pass 12
Exact edit unchanged: validated K0 + outer(u,v), lambda .001; other tensors original. READ edited on same 80 single-cue counterfactual requests in counterfactual_edited.json. Expected response predicted from pass 11 cached head inputs, now verify actual edited output.

Pass 12 verifies counterfactual head effects, max centered-logit prediction error 1.69277e-05. The validated edit amplifies relevant cues: delivery exceptions 0.077->1.448, due date 1.155->4.758, constraint violations .728->3.157 mean signed log-odds. Allowlist .114->.378, but 1/10 edited responses reverses, a known limitation.

Query candidate: closed-form residual ridge in 64 decision PCs (plus intercept) x 16 option-key PCs, key output basis projected orthogonal to mean-query u. Subspaces from fit activations only. Targets residual log-gold contrasts after validated rank1, no gradients/iterative fitting.
- q lambda=1.0, strength=0.5, fit=0.8019, dev=0.7522, gain=+0.1744 SE=0.0215, Brier=0.0691
- q lambda=1.0, strength=1.0, fit=0.8085, dev=0.7511, gain=+0.1733 SE=0.0223, Brier=0.0721
- q lambda=1.0, strength=1.5, fit=0.8019, dev=0.7356, gain=+0.1578 SE=0.0225, Brier=0.0803
- q lambda=10.0, strength=0.5, fit=0.7996, dev=0.7578, gain=+0.1800 SE=0.0214, Brier=0.0690
- q lambda=10.0, strength=1.0, fit=0.8081, dev=0.7533, gain=+0.1756 SE=0.0218, Brier=0.0709
- q lambda=10.0, strength=1.5, fit=0.8033, dev=0.7378, gain=+0.1600 SE=0.0223, Brier=0.0772
- q lambda=100.0, strength=0.5, fit=0.7944, dev=0.7500, gain=+0.1722 SE=0.0218, Brier=0.0695
- q lambda=100.0, strength=1.0, fit=0.8037, dev=0.7500, gain=+0.1722 SE=0.0220, Brier=0.0691
- q lambda=100.0, strength=1.5, fit=0.8004, dev=0.7444, gain=+0.1667 SE=0.0225, Brier=0.0716
- q lambda=1000.0, strength=0.5, fit=0.7852, dev=0.7489, gain=+0.1711 SE=0.0216, Brier=0.0710
- q lambda=1000.0, strength=1.0, fit=0.7919, dev=0.7467, gain=+0.1689 SE=0.0219, Brier=0.0697
- q lambda=1000.0, strength=1.5, fit=0.7911, dev=0.7467, gain=+0.1689 SE=0.0223, Brier=0.0692

Query residual candidate best development +.0100 over validated key, paired SE 0.007436885592708706; no query weights saved. Not yet sufficient evidence to add complexity.

Rank-one refinement uses the same fit-only eigensystem, predeclared floors .003/.01/.03/.1, ridge .0001/.0003/.001/.003/.01, strengths .75/1. No test/validation used to choose. Best development candidate=(0.1, 0.0003, 0.75, (0.7911111116409302, 0.2155555933713913, 0.010437639483360293, 0.07090549916028976), (0.7722222208976746, 0.1944444626569748, 0.0185395086686121, 0.0873011127114296)). Still no new edit saved.

## Before pass 13
Exact edit: k.weight = K0 + outer(u,v_ref); v_ref is .75 times fit-only ridge solution lambda .0003 with log-gold floor .1, saved in candidate_rank1_refined/derivation.pt. All other tensors original. Refined candidate fixed using fit/dev only. Paired DECIDE on reserved 180 practice-test cases in paired_test.json. Previous test was used only to confirm best_1; its individual errors were not inspected or used in refinement.

Pass 13 refined rank1 practice-test: .591111 -> .747778, gain .156667, paired SE .019381; 99 improved, 25 worsened. Increment over best_1=0.02111111111111112, paired SE=0.011948287315068714.

## Before pass 14
Exact edit: k.weight = K0 + outer(u,v_ref), floor .1, ridge .0003, strength .75. All other tensors original. Hidden VALIDATION both, all 300 cases. Candidate chosen on fit/dev, not validation. best_1 remains fallback.

Pass 14 refined rank1 hidden validation: .582000 -> .734000, gain .152000, paired SE .014019; 174 improved, 35 worsened. Best confirmed score improves over best_1.

Current best confirmed edit backed up at best_2/edited. Investigate rank>1 key repair only if fit/dev supports it: residual query modes may condition the option-evidence readout by question context.

Joint key-readout experiment: one closed-form solve from ORIGINAL-head residual targets, not iterative residual fitting. Features = full 2048-dimensional mean-query key path plus 4 orthogonal query context modes x 256 option PCs; fit-only subspaces, log-gold floor .1. All candidates derived independently from same sufficient statistics. Results:
- lambda=0.0003, strength=0.5, fit=0.7644, dev=0.7500, inc_best2=-0.0222 SE=0.0106, Brier=0.1025
- lambda=0.0003, strength=0.75, fit=0.8185, dev=0.7744, inc_best2=+0.0022 SE=0.0097, Brier=0.0849
- lambda=0.0003, strength=1.0, fit=0.8404, dev=0.7767, inc_best2=+0.0044 SE=0.0109, Brier=0.0836
- lambda=0.001, strength=0.5, fit=0.7537, dev=0.7478, inc_best2=-0.0244 SE=0.0105, Brier=0.1027
- lambda=0.001, strength=0.75, fit=0.8011, dev=0.7767, inc_best2=+0.0044 SE=0.0088, Brier=0.0835
- lambda=0.001, strength=1.0, fit=0.8219, dev=0.7789, inc_best2=+0.0067 SE=0.0106, Brier=0.0793
- lambda=0.003, strength=0.5, fit=0.7448, dev=0.7456, inc_best2=-0.0267 SE=0.0108, Brier=0.1039
- lambda=0.003, strength=0.75, fit=0.7863, dev=0.7711, inc_best2=-0.0011 SE=0.0095, Brier=0.0838
- lambda=0.003, strength=1.0, fit=0.8085, dev=0.7756, inc_best2=+0.0033 SE=0.0100, Brier=0.0780
- lambda=0.01, strength=0.5, fit=0.7337, dev=0.7333, inc_best2=-0.0389 SE=0.0121, Brier=0.1066
- lambda=0.01, strength=0.75, fit=0.7719, dev=0.7722, inc_best2=-0.0000 SE=0.0091, Brier=0.0863
- lambda=0.01, strength=1.0, fit=0.7859, dev=0.7689, inc_best2=-0.0033 SE=0.0099, Brier=0.0791

## Before pass 15
Exact edit: k.weight = K0 + deltaK_joint; rank <=5, deltaK_joint in candidate_joint_key/derivation.pt. Joint closed-form lambda .001, strength1, fit540 only, floor .1. All other tensors original. Paired DECIDE reserved180 test IDs in paired_test.json. Development gain over best2 only .006667 SE .010559; test will determine whether extra context paths merit validation.

Pass 15 rank<=5 key practice-test: original .591111 -> edited .744444, gain .153333, SE .021205. Increment over best2 -0.003333, paired SE 0.011195. Extra context paths rejected; restored exact best_2/edited/k.weight.npy. Current edit is again validated rank-one K0+outer(u,v_ref), all other tensors original.

Geometry of best2: ||K0||=24.6776, ||deltaK||=261.2993; cos(deltaK,K0)=-.000954, rank1 reconstruction error <2.4e-7. Ten normalized option dimensions carry 89.987% of activation energy but only .00434% of v squared norm. Large relative parameter change compensates for low feature amplitude, not global scaling.
Best2 cached dev ablation top10_only: {'accuracy': 0.5777777894503541, 'gain': 0.0, 'se': 0.003859738279646906, 'brier': 0.16686628758907318}
Best2 cached dev ablation other2038_only: {'accuracy': 0.7711111207803091, 'gain': 0.19333333753877216, 'se': 0.018096918631288664, 'brier': 0.0777728334069252}
Best2 cached dev ablation schema_mean_span_only: {'accuracy': 0.6433333453204897, 'gain': 0.06555555711189906, 'se': 0.014851264210731217, 'brier': 0.1261548101902008}
Best2 cached dev ablation schema_mean_null_only: {'accuracy': 0.6988888999654187, 'gain': 0.12111111332972845, 'se': 0.01603048203203211, 'brier': 0.1165437400341034}

Caution: correction() includes option-common offsets; RMS values printed in run28 were NOT centered. Only accuracy and paired gain from that run are used; Brier must respect per-type temperature.

## Before pass 16
Exact edit: k.weight = K0 + outer(u,v_top10), v_top10 retains only dims [98,1816,1035,893,1855,281,1630,2,636,456] of validated v_ref; all other tensors original. Directions in ablation_directions.pt. Paired DECIDE same 180 development cases in paired_dev.json. Causal ablation only; best2 remains the saved validated choice. Cached prediction: accuracy .577778, gain 0.

Pass 16 actual top10-only ablation: development .577778 -> .577778, gain 0, SE .003860; 6 items improved and 6 worsened. Confirms cached prediction.
Corrected best2 temperature-aware dev statistics: {'accuracy': 0.7722222317424086, 'gain': 0.19444444866644012, 'se': 0.018539508509868297, 'brier': 0.08730112761259079}

## Before pass 17
Exact edit: k.weight = K0 + outer(u,v_other2038), v_other2038 equals validated v_ref with the ten highest-energy option dimensions zeroed. All other tensors original. Paired DECIDE 180 development cases in paired_dev.json. Prediction .771111 accuracy versus .772222 full repair. Diagnostic only; best2 unchanged on disk.

Pass 17 actual low-energy-only ablation: development .577778 -> .771111, gain .193333, SE .018097; 117 improved, 22 worsened. Full best2 is .772222 cached. Thus low-energy option-state readout accounts for essentially all benefit. Restored exact best2.
Final planned fit-only closed-form target comparison, same rank1 mechanism and no new subspaces: log(p+.05), log(p+.1), log(p+.2), and centered 4*p. Ridges .0003/.001 and strengths .75/1 fixed in advance; select on dev only, advance only if improvement over best2 is larger than paired dev uncertainty.

Smooth-target closed-form development comparisons (no saved weight changes):
- ('4*p', 0.001, 0.75): dev=0.782222, inc_best2=+0.010000 SE=0.007268, Brier=0.078682
- ('4*p', 0.0003, 0.75): dev=0.775556, inc_best2=+0.003333 SE=0.006587, Brier=0.078082
- ('log_p+.2', 0.001, 1.0): dev=0.772222, inc_best2=+0.000000 SE=0.009188, Brier=0.105058
- ('4*p', 0.001, 1.0): dev=0.768889, inc_best2=-0.003333 SE=0.008840, Brier=0.074895
- ('log_p+.05', 0.0003, 0.75): dev=0.766667, inc_best2=-0.005556 SE=0.006387, Brier=0.081117
- ('log_p+.1', 0.001, 1.0): dev=0.764444, inc_best2=-0.007778 SE=0.009502, Brier=0.084979
- ('4*p', 0.0003, 1.0): dev=0.763333, inc_best2=-0.008889 SE=0.009298, Brier=0.077209
- ('log_p+.2', 0.0003, 1.0): dev=0.763333, inc_best2=-0.008889 SE=0.009562, Brier=0.103960
- ('log_p+.1', 0.0003, 1.0): dev=0.762222, inc_best2=-0.010000 SE=0.009225, Brier=0.084853
- ('log_p+.1', 0.0003, 0.75): dev=0.762222, inc_best2=-0.010000 SE=0.005291, Brier=0.090035
- ('log_p+.2', 0.0003, 0.75): dev=0.761111, inc_best2=-0.011111 SE=0.005837, Brier=0.105303
- ('log_p+.05', 0.001, 1.0): dev=0.760000, inc_best2=-0.012222 SE=0.009606, Brier=0.074950
- ('log_p+.05', 0.001, 0.75): dev=0.758889, inc_best2=-0.013333 SE=0.007973, Brier=0.082397
- ('log_p+.05', 0.0003, 1.0): dev=0.758889, inc_best2=-0.013333 SE=0.009533, Brier=0.075926
- ('log_p+.1', 0.001, 0.75): dev=0.754444, inc_best2=-0.017778 SE=0.006552, Brier=0.091769
- ('log_p+.2', 0.001, 0.75): dev=0.750000, inc_best2=-0.022222 SE=0.007539, Brier=0.107252

Selected final target candidate using development only: target 4*p, ridge .001, strength .75. Small dev increment .010000 SE .007268 over best2; not asserted significant. Fit-only derivation remains rank1, independent solve, no training. Cosine with best2 0.936399.

## Before pass 18
Exact edit: k.weight = K0 + outer(u,v_linear), v_linear is .75 times ridge(lambda=.001) solve for centered [4*p - original logits/temperature], all other tensors original. Derivation in candidate_linear_rank1/derivation.pt. Paired DECIDE reserved 180 practice test cases in paired_test.json. No individual test errors used; best2 remains saved fallback.

Pass 18 linear-target rank1 reserved practice: original .591111 -> edited .746667, gain .155556, paired SE .020028; 103 improved, 29 worsened. Increment over best2 -0.001111, paired SE 0.007799; candidates effectively tied, no further adjustment from test results.

## Before pass 19
Exact edit: k.weight = K0 + outer(u,v_linear), v_linear in candidate_linear_rank1/derivation.pt, linear target 4*p, ridge .001, strength .75. Other tensors original. Hidden VALIDATION both on all 300 cases; no records visible. Candidate fixed before validation; best2 remains fallback.

Pass 19 linear-target hidden validation: original .582000 -> edited .727333, gain .145333, paired SE .014654; 168 improved, 39 worsened. Best2 remains the best confirmed candidate (.734000 validation). No more fitting or candidate selection; remaining passes are diagnostics and final verification.

## Before pass 20
Exact diagnostic edit: k.weight = K0 + .5*outer(u,v_ref), where v_ref is the fixed best2 direction (log-gold floor .1, ridge .0003, fitted strength .75). All other tensors original. Paired DECIDE on the same 180 development cases in paired_dev.json. Dose response uses multipliers 0/.5/1/1.5 of this fixed direction, not refitting. Final choice remains full-strength best2.

Pass 20 actual best2 half-dose development: original .577778 -> edited .697778, gain .120000, paired SE .014321; 86 improved, 15 worsened. Matches cached-head prediction exactly.

## Before pass 21
Exact diagnostic edit: k.weight = K0 + 1.5*outer(u,v_ref), using the fixed best2 direction; all other tensors original. Paired DECIDE same 180 development cases in paired_dev.json. Cached prediction .746667, below full-strength .772222. No refitting or candidate selection; full-strength best2 remains final choice.

Pass 21 actual best2 1.5-dose development: original .577778 -> edited .746667, gain .168889, paired SE .022218; 107 improved, 34 worsened. Matches cached-head prediction. Restored exact full-strength best2.
Frozen final k.weight SHA256: b2462ebda90afd3ca91ab230fdf4fc15ab2c845c0c003abd99daf1970cd0b5ca. Only edited tensor is k.weight.

## Before pass 22
Exact edit: restored best_2/edited/k.weight.npy, k.weight = K0 + outer(u,v_ref), log-gold floor .1, ridge .0003, strength .75. All other tensors original. Paired DECIDE 180 development cases in paired_dev.json. Full-strength actual dose-response confirmation; predicted .772222. Weights are frozen for all remaining passes.

Pass 22 final full-strength development: original .577778 -> edited .772222, gain .194444, paired SE .018540; 115 improved, 21 worsened. Actual fixed-direction dose response 0/.5/1/1.5 = .577778/.697778/.772222/.746667.
Final-edit predicted cue response delivery_exceptions: original/final means [0.07690327167510987, 0.8105327472090721], positive pair counts [10, 10]/10.
Final-edit predicted cue response payment_deadline: original/final means [1.1550676852464676, 2.0772709369659426], positive pair counts [10, 10]/10.
Final-edit predicted cue response constraint_violation: original/final means [0.7277813605964184, 1.3528353787958622], positive pair counts [10, 10]/10.
Final-edit predicted cue response source_allowlisted: original/final means [0.11406782269477844, 0.23764198422431945], positive pair counts [10, 9]/10.

## Before pass 23
Exact frozen edit: best2 k.weight = K0+outer(u,v_ref); all other tensors original. READ edited on the same 80 single-cue counterfactual requests (400 questions) in counterfactual_edited.json, paired definitions in counterfactual_pairs.json. All derived from fit examples, no synthetic labels used for fitting. Compare with pass11 original head inputs; verify final edit's causal cue responses.

Run39 diagnostic assertion encountered NaNs from padded -inf logits, not from model outputs. Fixed centering by excluding padding before summation; all valid-option logits are finite.

Pass 23 final counterfactual READ: max centered-logit prediction error 1.1444092e-05. Actual responses verify stronger sensitivity to delivery exceptions, due dates, and constraint violations (all 10/10 pairs in expected direction); allowlisting 9/10, retained limitation. No weight changes.
- delivery_exceptions: original/final response means [0.07690327167510987, 0.8105318494141102], expected signs [10, 10]/10.
- payment_deadline: original/final response means [1.1550676852464676, 2.07727056145668], expected signs [10, 10]/10.
- constraint_violation: original/final response means [0.7277813605964184, 1.3528353277593852], expected signs [10, 10]/10.
- source_allowlisted: original/final response means [0.11406782269477844, 0.23764199614524842], expected signs [10, 9]/10.

## Before pass 24
Exact frozen edit: best_2/edited/k.weight.npy, K0+outer(u,v_ref); all other tensors original. DECIDE edited on all 900 practice IDs in final_all_practice.json; compare itemwise against original pass1. This is a final audit only: fit540/dev180/test180 remain separately reported, and no further tuning will follow.

Pass 24 frozen best2 final practice audit (no further tuning):
- all_practice: {"items": 900, "original": 0.5791111111111111, "edited": 0.7786666666666666, "gain": 0.19955555555555557, "paired_se": 0.008271727957570694, "improved": 571, "worsened": 90}
- fit: {"items": 540, "original": 0.5755555555555556, "edited": 0.7911111111111111, "gain": 0.21555555555555556, "paired_se": 0.01043763950652292, "improved": 357, "worsened": 44}
- dev: {"items": 180, "original": 0.5777777777777777, "edited": 0.7722222222222223, "gain": 0.19444444444444445, "paired_se": 0.01853950807173797, "improved": 115, "worsened": 21}
- test: {"items": 180, "original": 0.5911111111111111, "edited": 0.7477777777777775, "gain": 0.15666666666666668, "paired_se": 0.019380536993403977, "improved": 99, "worsened": 25}
- agent_trace_observability: {"items": 225, "original": 0.5235555555555556, "edited": 0.768, "gain": 0.24444444444444444, "paired_se": 0.015958499971500704, "improved": 156, "worsened": 13}
- customer_service: {"items": 225, "original": 0.6026666666666668, "edited": 0.8044444444444444, "gain": 0.20177777777777778, "paired_se": 0.016426416544512292, "improved": 150, "worsened": 27}
- invoice_processing: {"items": 225, "original": 0.5493333333333333, "edited": 0.7893333333333334, "gain": 0.24, "paired_se": 0.017593288763724922, "improved": 157, "worsened": 21}
- security_incidents: {"items": 225, "original": 0.6408888888888888, "edited": 0.7528888888888889, "gain": 0.11200000000000002, "paired_se": 0.014623312546795608, "improved": 108, "worsened": 29}

## Before pass 25
Exact final edit unchanged: best_2/edited/k.weight.npy, k.weight = K0 + outer(u,v_ref); log-gold floor .1, closed-form ridge .0003, strength .75, fit540 only. All other tensors original. SHA256 b2462ebda90afd3ca91ab230fdf4fc15ab2c845c0c003abd99daf1970cd0b5ca. Final hidden VALIDATION both on all 300 cases. This repeats pass14 to verify the restored submission weights, not an independent statistical replication. No more fitting, selection, or weight changes are planned.
