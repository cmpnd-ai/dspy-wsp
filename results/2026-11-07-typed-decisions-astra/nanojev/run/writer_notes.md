# NanoJev targeted weight-repair log
Initial state: no edits made. Inspecting schema, then original-model errors. Rules: closed-form head analysis only; no backbone execution, gradients, or iterative training.

Planned forward pass 1: original model, all 900 practice IDs in trainset order. Exact current edit: []

Forward 1 original all 900: score 0.364222, item SE 0.007638. 890/900 items imperfect. Next: inspect per-question failures and decompose head logits.

Question accuracy/frequencies:
agent_trace_observability/action: 3/225; pred={'stop': 217, 'human_review': 6, 'observe': 2}; gold={'human_review': 104, 'continue': 32, 'observe': 87, 'stop': 2}
agent_trace_observability/needs_review: 104/225; pred={'true': 225}; gold={'true': 104, 'false': 121}
agent_trace_observability/outcome: 49/225; pred={'partial': 8, 'success': 170, 'harmful': 47}; gold={'partial': 108, 'success': 59, 'harmful': 45, 'failure': 13}
agent_trace_observability/risk: 57/225; pred={'3': 8, '0': 203, '1': 9, '2': 5}; gold={'0': 56, '3': 45, '1': 77, '2': 47}
agent_trace_observability/urgency: 24/225; pred={'2': 196, '3': 29}; gold={'1': 114, '0': 54, '3': 39, '2': 18}
customer_service/action: 106/225; pred={'escalate_to_human': 157, 'answer_directly': 53, 'execute_refund': 13, 'request_information': 2}; gold={'answer_directly': 48, 'escalate_to_human': 127, 'request_information': 44, 'execute_refund': 6}
customer_service/category: 55/225; pred={'billing': 205, 'account': 6, 'delivery': 2, 'refund': 12}; gold={'account': 51, 'refund': 19, 'technical': 58, 'delivery': 48, 'billing': 49}
customer_service/churn_risk: 59/225; pred={'1': 224, '3': 1}; gold={'3': 52, '1': 58, '0': 11, '2': 104}
customer_service/needs_human: 152/225; pred={'true': 225}; gold={'true': 152, 'false': 73}
customer_service/urgency: 92/225; pred={'2': 220, '3': 5}; gold={'3': 89, '2': 90, '1': 45, '0': 1}
invoice_processing/discrepancy_severity: 55/225; pred={'2': 225}; gold={'3': 70, '0': 99, '2': 55, '1': 1}
invoice_processing/disposition: 78/225; pred={'manual_review': 135, 'approve': 89, 'hold': 1}; gold={'manual_review': 86, 'hold': 40, 'reject': 12, 'approve': 87}
invoice_processing/duplicate: 21/225; pred={'true': 225}; gold={'false': 204, 'true': 21}
invoice_processing/matches_order: 108/225; pred={'true': 225}; gold={'false': 117, 'true': 108}
invoice_processing/urgency: 101/225; pred={'2': 225}; gold={'2': 101, '3': 31, '1': 93}
security_incidents/credential_compromise: 120/225; pred={'true': 225}; gold={'true': 120, 'false': 105}
security_incidents/disposition: 162/225; pred={'investigate': 223, 'contain': 2}; gold={'investigate': 163, 'contain': 24, 'monitor': 17, 'close_benign': 21}
security_incidents/severity: 72/225; pred={'2': 220, '3': 5}; gold={'1': 31, '3': 97, '2': 68, '0': 5, '4': 24}
security_incidents/true_positive: 154/225; pred={'true': 225}; gold={'true': 154, 'false': 71}
security_incidents/urgency: 67/225; pred={'3': 93, '2': 9, '1': 123}; gold={'1': 44, '2': 59, '3': 116, '0': 6}

Fixed split seed 20260202: fit450/dev150/test300, stratified by workflow; IDs saved in analysis_splits.json. No fit or hyperparameter decisions will use test labels.
Planned forward 2: original READ first 200 fit IDs (exact list read2_request.json); 1000 questions. Edits: none. Inspect EOS states, normalized features, scalar and set correction.

Planned forward 3: original READ fit IDs 200:400 (exact read3_request.json), 1000 questions. Edits: none. Complete evidence across all workflows before deriving a readout direction.

Planned forward 4: original READ last50 fit IDs + all150 dev IDs, exact read4_request.json. Edits: none. Boolean false fixed-logit slot must be excluded from scalar reconstruction. Head decomposition and closed-form design follow.

Read alignment now verified: source-file order, max probability error zero. Scalar correction design: delta=(D'D+lambda I)^-1 D'(center(log gold)-center(original logits)); center features per question, fixed false feature zero, equal question weight. Fit450 only; dev150 used for ridge/strength choice. Original set-attention retained for preliminary cached evaluation; exact recomputation next. No edits saved yet.

Diagnosis: no-set-attention dev accuracy 0.370667 vs original 0.373333, so set attention is not main defect. Scalar closed-form log-probability correction ridge0.01: fit0.638667/dev0.545333 (cached original set correction); norm100.35, cosine with original scalar0.0051. Corrections vary over records (~0.5–2 logit SD), including category and severity cues. Exploring bounded contrasts as an independently solved closed-form alternative, not iterative loss fitting.

Closed-form soft direction: fit450, centered targets 3*gold probability, ridge0.01, delta norm68.7146. Chosen as main candidate for lower norm and better Brier than hard/mixed targets. Exact-form head reconstruction residual checked above. Planned forward5: paired DECIDE all150 dev IDs, exact dev_decide_request.json. Current exact edit: scalar.weight=w_original+0.5*d_soft; d_soft saved candidate_weights/soft_direction.pt. All other tensors original. This is a directional readout change, not global scaling.

Forward5 actual half-strength soft direction on dev150: original0.373333, edited0.520000, paired gain0.146667 SE0.018114, 82 improved/19 worsened. Matches cached head prediction exactly; causal evidence for scalar direction.

Planned forward6: paired DECIDE same150 dev IDs, dev_decide_request.json. Exact current edit: scalar.weight=w_original+1.0*d_soft (candidate_weights/soft_full_scalar.pt). All other tensors original. Prediction: dev~0.5853, larger gain than strength0.5.

Forward6 full-strength soft direction dev150: original0.373333, edited0.585333, gain0.212000 SE0.025329, 96 improved/24 worsened. Dose response actual: strength0 ->37.33%, .5 ->52.00%, 1 ->58.53%.

Planned forward7: paired DECIDE reserved test300 IDs (test_decide_request.json), never used to fit or choose strength. Exact current edit scalar.weight=w_original+d_soft, saved candidate_weights/soft_full_scalar.pt; all other tensors original. Candidate frozen before this independent measurement.

Forward7 independent practice test300: original0.358000, edited0.556667, paired gain0.198667 SE0.016027; 189 improved,44 worsened. Candidate was fit/selected without these300.

Planned forward8: paired hidden VALIDATION300. Exact current edit scalar.weight=w_original+d_soft (candidate_weights/soft_full_scalar.pt), no other changes. No validation records or labels will be available.

Forward8 hidden validation300 confirmed soft-full: original0.348000, edited0.560000, paired gain0.212000 SE0.015787;199 improved/42 worsened. Current best confirmed candidate: soft-full.

Soft direction decomposition, fit-only mean option-template span rank51: prior norm8.3082, evidence norm68.2105. Full candidate backed up to best_1 and best_1_weights.pt. Next controlled prior-only ablation and matched-state cue tests.

Planned forward9: paired DECIDE dev150, same dev_decide_request.json. Exact edit scalar.weight=w_original+d_prior (projection of validated d_soft onto fit mean option-template span); all others original. Tests whether full gain can be explained by question/option priors alone. Restore best1 afterward.

Forward9 prior-only dev150:0.480000 vs original0.373333, gain0.106667 SE0.022118. Full direction0.585333 on same records. Record-specific component contributes beyond template correction.

Planned forward10: original READ controlled cue requests, exact cue_original_request.json (64 requests,100 questions; manifest cue_manifest.json). Current edit restored scalar.weight=w_original+d_soft. Synthetic variations are diagnostic only, not fitting labels: violations0/1/3; five customer topics; days_due30/7/0/-5; asset criticality low/medium/high/critical. Prediction: edited head has stronger correctly directed responses, especially agent violations and customer category.

Planned forward11: edited READ exact same64 controlled requests/100 questions as forward10, cue_edited_request.json. Exact edit scalar.weight=w_original+d_soft, all others original. Compare raw margins, true probabilities, and score expectations with original; these synthetic records are never used to fit weights.

Cue test forward10/11 (same hidden features):
violations/action/0: ['continue', 'human_review', 'observe', 'stop']; original [0.1459999978542328, 0.19200000166893005, 0.1940000057220459, 0.46799999475479126]; edited [0.20200000703334808, 0.36899998784065247, 0.29499998688697815, 0.13300000131130219]
violations/needs_review/0: P(true) 0.804->0.330
violations/outcome/0: ['failure', 'harmful', 'partial', 'success']; original [0.1459999978542328, 0.2720000147819519, 0.27000001072883606, 0.31200000643730164]; edited [0.18400000035762787, 0.16099999845027924, 0.3880000114440918, 0.2680000066757202]
violations/risk/0: expected level 1.375->1.309
violations/action/1: ['continue', 'human_review', 'observe', 'stop']; original [0.12999999523162842, 0.18299999833106995, 0.17599999904632568, 0.5109999775886536]; edited [0.210999995470047, 0.3659999966621399, 0.2809999883174896, 0.14100000262260437]
violations/needs_review/1: P(true) 0.779->0.613
violations/outcome/1: ['failure', 'harmful', 'partial', 'success']; original [0.14100000262260437, 0.28200000524520874, 0.27300000190734863, 0.30399999022483826]; edited [0.2029999941587448, 0.21299999952316284, 0.3330000042915344, 0.25099998712539673]
violations/risk/1: expected level 1.404->1.540
violations/action/3: ['continue', 'human_review', 'observe', 'stop']; original [0.1289999932050705, 0.17900000512599945, 0.1679999977350235, 0.5239999890327454]; edited [0.20100000500679016, 0.33000001311302185, 0.2759999930858612, 0.19300000369548798]
violations/needs_review/3: P(true) 0.784->0.461
violations/outcome/3: ['failure', 'harmful', 'partial', 'success']; original [0.1459999978542328, 0.27000001072883606, 0.2639999985694885, 0.3199999928474426]; edited [0.25699999928474426, 0.16699999570846558, 0.3190000057220459, 0.25699999928474426]
violations/risk/3: expected level 1.377->1.407
topic/account: target P 0.229->0.282; top original=billing, edited=technical
topic/billing: target P 0.369->0.615; top original=billing, edited=billing
topic/delivery: target P 0.282->0.679; top original=billing, edited=delivery
topic/refund: target P 0.184->0.331; top original=billing, edited=refund
topic/technical: target P 0.155->0.531; top original=billing, edited=technical
days_due/urgency/30: expected level 1.864->1.496
days_due/urgency/7: expected level 1.872->1.501
days_due/urgency/0: expected level 1.842->1.436
days_due/urgency/-5: expected level 1.858->1.490
asset_criticality/severity/low: expected level 2.361->2.301
asset_criticality/severity/medium: expected level 2.368->2.319
asset_criticality/severity/high: expected level 2.434->2.485
asset_criticality/severity/critical: expected level 2.395->2.361

Full minus prior-only on development150: gain0.105333 SE0.021463. Thus roughly half the full development gain requires the evidence component, not template priors alone.

Cue diagnosis: head inputs unchanged exactly under edit. Topic target probabilities improved account .229->.282, billing .369->.615, delivery .282->.679, refund .184->.331, technical .155->.531. Violations0->1 needs_review response changes original .804->.779 (wrong direction) to edited .330->.613 (right). Violations3 and invoice deadlines remain unreliable; no claim of complete numeric repair.

Forward12 plan: paired DECIDE reserved test300, test_decide_request.json. Candidate mix600_075: closed-form centered target3*(.5gold+.5top-label), fit all600 fit+dev only; ridge=.004 (same per-item regularization as .003 on450); strength=.75 selected on earlier dev before refit. Exact edit scalar.weight=w_original+.75*d_mix600, saved candidate_weights/mix600_075_scalar.pt; all other tensors original. Best validated soft-full safely backed up best_1. No independent test labels used in this new solve or hyperparameter choice.

Forward12 mix600_075 independent practice300: original0.358000, edited0.568667, paired gain0.210667 SE0.014184;202 improved/29 worsened.

mix600 vs prior soft-full on independent practice300: 0.012000 SE0.010620; small incremental difference, not claimed significant.

Planned forward13: paired hidden VALIDATION300. Exact current edit scalar.weight=w_original+.75*d_mix600, candidate_weights/mix600_075_scalar.pt. All others original. Best1 still preserved.

Forward13 mix600_075 hidden validation300: original0.348000, edited0.574000, gain0.226000 SE0.014991;207 improved/32 worsened. New best confirmed candidate; its +1.4pp vs best1 on validation is not claimed statistically significant because only aggregate results are available.

Planned forward14: edited READ only choice questions for fit450+dev150, exact choice600_read_request.json and choice600_manifest.json, 900 questions. Exact edit scalar.weight=w_original+.75*d_mix600 (best2), all other tensors original. Purpose: measure actual set correction under validated edit before any set-head intervention.

Forward14 verified scalar-independent set correction; exact set formula: tanh(project([head_input,log(option_count)]) + four-head self-attention output), then set_output. Testing fit450-only residual/joint closed-form set-output edits on dev150; no test or validation labels used.

Set head residual/joint fits yield at most dev0.605333 vs mixed scalar0.600000, too small to justify a second component. No set edits saved. Testing prespecified one-shot Fisher/top-margin weighted contrasts using fit450/dev150 only.

Planned forward15: original READ first200 test IDs, exact read15_request.json,1000 questions. Current edit unchanged best2 scalar.weight=w_original+.75*d_mix600; original mode ignores it. These300 will be retired as independent data when used for full900 refit; prior independent practice results remain valid for frozen earlier candidates. New recipe selection remains fit450/dev150 only.

Forward15 read200 completed. Weighted contrasts did not materially beat simpler mixed target (.601333 vs .600000 dev). Keep simple recipe; no weighted or set-output edits saved.

Planned forward16: original READ final100 test IDs, exact read16_request.json,500 questions. Current edit best2 scalar.weight=w_original+.75*d_mix600; no other tensors. Purpose: complete900 features for cross-fit measurement and fixed-recipe refit. Full900 candidate will require fresh hidden-validation confirmation.


Forward16 completed final100 original READs. All900 practice features are now available. The earlier test300 is retired for new fitting; frozen-candidate forward7/12 estimates remain independent. Next: fixed-recipe full900 solve, not hyperparameter selection on test labels.


Fixed full900 mixed-target recipe reproduced600 direction (numerical tolerance verified). lambda=.006, strength=.75. Full900 in-sample: {'accuracy': 0.2611111111111111, 'gain': -0.10311111111111111, 'paired_se': 0.013305091005153098, 'brier_sum': nan, 'items': 900}. Fixed-recipe three-fold estimate (not an untouched recipe-selection set): {'accuracy': 0.2611111111111111, 'gain': -0.10311111111111111, 'paired_se': 0.013305091005153098, 'brier_sum': nan, 'items': 900}. No new edits saved yet; best2 remains active.


CORRECTION: preceding full900/cross-fit numbers were invalid NaN artifacts from multiplying read-file NaN padding by zero. No invalid edit was installed. Padding now zeroed with np.where; all actual path features finite. Valid full900 in-sample {'accuracy': 0.6537777777777778, 'gain': 0.28955555555555557, 'paired_se': 0.008456583719765088, 'brier_sum': 0.15720486640930176, 'items': 900}; valid fixed-recipe cross-fit {'accuracy': 0.568, 'gain': 0.2037777777777778, 'paired_se': 0.00893453549730807, 'brier_sum': 0.19290438294410706, 'items': 900}.


Planned forward17: paired hidden VALIDATION300. Exact edit scalar.weight=w_original+.75*d_mix900, candidate_weights/mix900_075_scalar.pt, lambda .006 fixed before viewing new features/labels; all other tensors original. No new recipe tuning. best2 backed up. All900 practice now used in solve; validation remains unseen.


Forward17 hidden validation300 full900 mix(.75): original .348000, edited .579333, paired gain .231333 SE .014716;205 improved/25 worsened. New best confirmed candidate; +.005333 over best2 not claimed significant.


Fixed-recipe cross-fit strength diagnostic (head only; no forward and no weight fitting): [{'strength': 0, 'accuracy': 0.3642222222222222, 'gain': 0.0, 'paired_se': 0.0, 'brier_sum': 0.328453004360199, 'items': 900}, {'strength': 0.25, 'accuracy': 0.4346666666666667, 'gain': 0.07044444444444445, 'paired_se': 0.004888298972285244, 'brier_sum': 0.2637629806995392, 'items': 900}, {'strength': 0.5, 'accuracy': 0.5297777777777778, 'gain': 0.16555555555555557, 'paired_se': 0.007341807742290631, 'brier_sum': 0.21544349193572998, 'items': 900}, {'strength': 0.625, 'accuracy': 0.5588888888888889, 'gain': 0.19466666666666665, 'paired_se': 0.008263848240345633, 'brier_sum': 0.20089109241962433, 'items': 900}, {'strength': 0.75, 'accuracy': 0.568, 'gain': 0.2037777777777778, 'paired_se': 0.00893453549730807, 'brier_sum': 0.19290438294410706, 'items': 900}, {'strength': 0.875, 'accuracy': 0.5715555555555555, 'gain': 0.20733333333333337, 'paired_se': 0.009343259724876065, 'brier_sum': 0.19097289443016052, 'items': 900}, {'strength': 1, 'accuracy': 0.5715555555555554, 'gain': 0.20733333333333337, 'paired_se': 0.009681084462307565, 'brier_sum': 0.19424958527088165, 'items': 900}, {'strength': 1.25, 'accuracy': 0.5542222222222222, 'gain': 0.19, 'paired_se': 0.009960371361134673, 'brier_sum': 0.21223032474517822, 'items': 900}]. best3 full900 strength.75 backed up.


Cross-fit decomposition diagnostic, finite predeclared strength grid: [{'prior_strength': 0.5, 'evidence_strength': 0.5, 'accuracy': 0.5297777777777778, 'gain': 0.16555555555555557, 'paired_se': 0.007341807742290631, 'brier_sum': 0.21544356644153595, 'items': 900}, {'prior_strength': 0.5, 'evidence_strength': 0.75, 'accuracy': 0.5311111111111111, 'gain': 0.16688888888888886, 'paired_se': 0.008042441560709834, 'brier_sum': 0.21702834963798523, 'items': 900}, {'prior_strength': 0.5, 'evidence_strength': 0.875, 'accuracy': 0.5326666666666666, 'gain': 0.16844444444444442, 'paired_se': 0.008430382035876084, 'brier_sum': 0.22037209570407867, 'items': 900}, {'prior_strength': 0.5, 'evidence_strength': 1, 'accuracy': 0.5282222222222221, 'gain': 0.16400000000000003, 'paired_se': 0.008750485976228437, 'brier_sum': 0.2251506894826889, 'items': 900}, {'prior_strength': 0.5, 'evidence_strength': 1.25, 'accuracy': 0.5182222222222223, 'gain': 0.15400000000000003, 'paired_se': 0.009090081871555978, 'brier_sum': 0.23815910518169403, 'items': 900}, {'prior_strength': 0.75, 'evidence_strength': 0.5, 'accuracy': 0.57, 'gain': 0.20577777777777775, 'paired_se': 0.008464481947988631, 'brier_sum': 0.19116796553134918, 'items': 900}, {'prior_strength': 0.75, 'evidence_strength': 0.75, 'accuracy': 0.568, 'gain': 0.2037777777777778, 'paired_se': 0.00893453549730807, 'brier_sum': 0.19290447235107422, 'items': 900}, {'prior_strength': 0.75, 'evidence_strength': 0.875, 'accuracy': 0.5637777777777777, 'gain': 0.19955555555555557, 'paired_se': 0.009156747741183928, 'brier_sum': 0.19661450386047363, 'items': 900}, {'prior_strength': 0.75, 'evidence_strength': 1, 'accuracy': 0.5628888888888889, 'gain': 0.19866666666666669, 'paired_se': 0.009311909306110405, 'brier_sum': 0.20190244913101196, 'items': 900}, {'prior_strength': 0.75, 'evidence_strength': 1.25, 'accuracy': 0.5524444444444444, 'gain': 0.1882222222222222, 'paired_se': 0.009511304123890401, 'brier_sum': 0.21619398891925812, 'items': 900}, {'prior_strength': 0.875, 'evidence_strength': 0.5, 'accuracy': 0.5746666666666668, 'gain': 0.21044444444444446, 'paired_se': 0.009000324237984685, 'brier_sum': 0.18591555953025818, 'items': 900}, {'prior_strength': 0.875, 'evidence_strength': 0.75, 'accuracy': 0.5735555555555556, 'gain': 0.20933333333333337, 'paired_se': 0.00932803493869824, 'brier_sum': 0.1873435378074646, 'items': 900}, {'prior_strength': 0.875, 'evidence_strength': 0.875, 'accuracy': 0.5715555555555555, 'gain': 0.20733333333333337, 'paired_se': 0.009343259724876065, 'brier_sum': 0.19097299873828888, 'items': 900}, {'prior_strength': 0.875, 'evidence_strength': 1, 'accuracy': 0.564, 'gain': 0.1997777777777778, 'paired_se': 0.009446422470202482, 'brier_sum': 0.19622164964675903, 'items': 900}, {'prior_strength': 0.875, 'evidence_strength': 1.25, 'accuracy': 0.5575555555555555, 'gain': 0.19333333333333333, 'paired_se': 0.00974016653593862, 'brier_sum': 0.21052749454975128, 'items': 900}, {'prior_strength': 1, 'evidence_strength': 0.5, 'accuracy': 0.5722222222222222, 'gain': 0.208, 'paired_se': 0.009318808550819273, 'brier_sum': 0.18492016196250916, 'items': 900}, {'prior_strength': 1, 'evidence_strength': 0.75, 'accuracy': 0.5733333333333334, 'gain': 0.2091111111111111, 'paired_se': 0.009472901648357922, 'brier_sum': 0.18582585453987122, 'items': 900}, {'prior_strength': 1, 'evidence_strength': 0.875, 'accuracy': 0.573111111111111, 'gain': 0.2088888888888889, 'paired_se': 0.00961558035689537, 'brier_sum': 0.18921969830989838, 'items': 900}, {'prior_strength': 1, 'evidence_strength': 1, 'accuracy': 0.5715555555555554, 'gain': 0.20733333333333337, 'paired_se': 0.009681084462307565, 'brier_sum': 0.19424965977668762, 'items': 900}, {'prior_strength': 1, 'evidence_strength': 1.25, 'accuracy': 0.5571111111111111, 'gain': 0.19288888888888892, 'paired_se': 0.009895916674988991, 'brier_sum': 0.20817381143569946, 'items': 900}, {'prior_strength': 1.25, 'evidence_strength': 0.5, 'accuracy': 0.5544444444444444, 'gain': 0.19022222222222224, 'paired_se': 0.009459105141205019, 'brier_sum': 0.1941998153924942, 'items': 900}, {'prior_strength': 1.25, 'evidence_strength': 0.75, 'accuracy': 0.5613333333333334, 'gain': 0.19711111111111113, 'paired_se': 0.009673478280349914, 'brier_sum': 0.19343598186969757, 'items': 900}, {'prior_strength': 1.25, 'evidence_strength': 0.875, 'accuracy': 0.5642222222222222, 'gain': 0.2, 'paired_se': 0.009625287010954943, 'brier_sum': 0.19594715535640717, 'items': 900}, {'prior_strength': 1.25, 'evidence_strength': 1, 'accuracy': 0.5664444444444444, 'gain': 0.20222222222222225, 'paired_se': 0.009742422117867992, 'brier_sum': 0.20008151233196259, 'items': 900}, {'prior_strength': 1.25, 'evidence_strength': 1.25, 'accuracy': 0.5542222222222222, 'gain': 0.19, 'paired_se': 0.009960371361134673, 'brier_sum': 0.2122303545475006, 'items': 900}]. Components are orthogonal projection onto training-fold mean centered question/option span and its complement; complement is not claimed to encode a single semantic cue. best3 unchanged.


Two additional candidates frozen before validation: A=w0+.875*d_template900+.5*d_orthogonal900; B=w0+.875*d_mix900. Chosen from cached practice cross-fit only; small differences among them are not independent performance estimates. No further hyperparameters will be selected from validation.


Planned forward18: paired hidden VALIDATION300. Exact edit scalar.weight=candidate_weights/mix900_decomposed_scalar.pt =w0+.875*dp900+.5*de900; all other tensors original. Full900 practice fit; candidate A selected on cross-fit. best3 strength.75 preserved.


Forward18 decomposed candidate hidden validation300: original .348000, edited .570667, paired gain .222667 SE .014754;206 improved/31 worsened. Lower score than best3(.579333); not retained.


Planned forward19: paired hidden VALIDATION300. Exact edit scalar.weight=w0+.875*d_mix900, candidate_weights/mix900_0875_scalar.pt; all other tensors original. Candidate B was fixed before forward18. After this, freeze best validation-confirmed candidate; remaining passes are practice verification, matched-cue diagnostics and a prior-only ablation, not further recipe selection.


Forward19 strength.875 hidden validation300: original .348000, edited .580667, paired gain .232667 SE .015184;207 improved/30 worsened. Best confirmed aggregate score. Difference +.001333 vs strength.75 is not claimed significant. Candidate selection now frozen.


Planned forward20: edited DECIDE all900 practice IDs in original trainset order, final_practice900_request.json. Exact edit frozen best4 scalar.weight=w0+.875*d_mix900 (candidate_weights/mix900_0875_scalar.pt); all other tensors original. Compare deterministic outputs with forward1 and cached full900 head calculation. This is in-sample verification, not independent evidence.


Forward20 final frozen best4 all900 fitted practice: original .364222, edited .669778; paired gain 0.3055555555555556 SE 0.00881652350231889. This is in-sample, not a held-out gain. Cached scalar-head prediction matches aggregate program accuracy.


Planned forward21: paired DECIDE dev150+retired-test300 (450 IDs in final_prior_ablation_request.json). Exact temporary edit scalar.weight=w0+.875*dp900, candidate_weights/mix900_prior_only_scalar.pt; all other tensors original. Mechanism-only ablation on fitted practice records, not independent accuracy evidence. Restore frozen best4 immediately afterward.


Forward21 prior-only ablation450: original .363111, prior-only .496889, paired gain .133778 SE .012328. Full frozen best4 on same450 .650222. Full minus prior-only {'gain': 0.15333333333333332, 'se': 0.011452320923767165, 'improved': 262, 'worsened': 63}. Fitted-record causal diagnostic; cross-fit full-minus-prior independently .070889 SE .008307. Restore best4 now.


Planned forward22: original READ matched synthetic requests, final_cue_original_request.json, 124 requests/268 questions. Manifest final_cue_manifest.json. Current edit restored frozen best4 scalar.weight=w0+.875*d_mix900; original mode ignores it. Cue tests include previous numeric/topic probes plus duplicate-ID, amount mismatch, authorization evidence and cancellation-intent variants. Diagnostic only, no new fitting.


Planned forward23: edited READ the exact same124 requests/268 questions as forward22, final_cue_edited_request.json. Exact edit frozen best4 scalar.weight=w0+.875*d_mix900, candidate_weights/mix900_0875_scalar.pt; all other tensors original. Verify identical head inputs and characterize directional cue responses, including failures. No weight fitting or selection.


Final matched-cue tests forward22/23: identical backbone/head-input/set-delta arrays; scalar reconstruction verified.
violations/action/0: P(stop) 0.468->0.159
violations/needs_review/0: P(true) 0.804->0.306
violations/outcome/0: P(harmful) 0.272->0.126
violations/risk/0: expected level 1.375->1.141
violations/action/1: P(stop) 0.511->0.251
violations/needs_review/1: P(true) 0.779->0.598
violations/outcome/1: P(harmful) 0.282->0.191
violations/risk/1: expected level 1.404->1.403
violations/action/3: P(stop) 0.524->0.229
violations/needs_review/3: P(true) 0.784->0.564
violations/outcome/3: P(harmful) 0.270->0.144
violations/risk/3: expected level 1.377->1.305
topic/category/account: P(account) 0.229->0.322; correct top 0/4 -> 4/4
topic/category/billing: P(billing) 0.369->0.642; correct top 4/4 -> 4/4
topic/category/delivery: P(delivery) 0.282->0.698; correct top 2/4 -> 4/4
topic/category/refund: P(refund) 0.184->0.400; correct top 0/4 -> 4/4
topic/category/technical: P(technical) 0.155->0.426; correct top 0/4 -> 4/4
days_due/urgency/30: expected level 1.864->1.472
days_due/urgency/7: expected level 1.872->1.471
days_due/urgency/0: expected level 1.842->1.482
days_due/urgency/-5: expected level 1.858->1.479
asset_criticality/severity/low: expected level 2.361->2.273
asset_criticality/severity/medium: expected level 2.368->2.350
asset_criticality/severity/high: expected level 2.434->2.528
asset_criticality/severity/critical: expected level 2.395->2.384
duplicate_id/duplicate/0: P(true) 0.778->0.243
duplicate_id/disposition/0: P(approve) 0.246->0.351
duplicate_id/duplicate/1: P(true) 0.785->0.193
duplicate_id/disposition/1: P(approve) 0.246->0.359
invoice_amount_ratio/matches_order/1.0: P(true) 0.773->0.751
invoice_amount_ratio/discrepancy_severity/1.0: expected level 1.798->1.058
invoice_amount_ratio/disposition/1.0: P(approve) 0.242->0.365
invoice_amount_ratio/matches_order/1.1: P(true) 0.762->0.647
invoice_amount_ratio/discrepancy_severity/1.1: expected level 1.812->1.045
invoice_amount_ratio/disposition/1.1: P(approve) 0.246->0.362
invoice_amount_ratio/matches_order/1.5: P(true) 0.765->0.616
invoice_amount_ratio/discrepancy_severity/1.5: expected level 1.803->1.138
invoice_amount_ratio/disposition/1.5: P(approve) 0.247->0.357
authorization_evidence/credential_compromise/authorized: P(true) 0.854->0.635
authorization_evidence/true_positive/authorized: P(true) 0.858->0.813
authorization_evidence/disposition/authorized: P(contain) 0.276->0.150
authorization_evidence/credential_compromise/compromised: P(true) 0.861->0.695
authorization_evidence/true_positive/compromised: P(true) 0.871->0.832
authorization_evidence/disposition/compromised: P(contain) 0.260->0.154
cancellation_intent/churn_risk/satisfied: expected level 1.615->1.562
cancellation_intent/needs_human/satisfied: P(true) 0.858->0.511
cancellation_intent/action/satisfied: P(escalate_to_human) 0.285->0.335
cancellation_intent/churn_risk/frustrated: expected level 1.614->1.595
cancellation_intent/needs_human/frustrated: P(true) 0.860->0.457
cancellation_intent/action/frustrated: P(escalate_to_human) 0.292->0.336
cancellation_intent/churn_risk/leaving: expected level 1.647->1.801
cancellation_intent/needs_human/leaving: P(true) 0.856->0.453
cancellation_intent/action/leaving: P(escalate_to_human) 0.313->0.386


Planned forward24: edited READ same124 matched requests/268 questions, final_cue_half_request.json. Exact temporary diagnostic edit scalar.weight=w0+.5*(best4-w0)=w0+.4375*d_mix900; all other tensors original. Not a candidate-selection run. Test dose response and restore best4 immediately afterward; final forward25 confirms frozen best4 on hidden validation.


Forward24 half-strength cue diagnostic: maximum midpoint-logit error 1.1555850505828857e-05. Strength0/half/full responses:
violations/0: P(true) 0.804->0.572->0.306
violations/1: P(true) 0.779->0.696->0.598
violations/3: P(true) 0.784->0.684->0.564
topic/account: target P 0.229->0.277->0.322; top counts [0, 1, 4]/4
topic/billing: target P 0.369->0.513->0.642; top counts [4, 4, 4]/4
topic/delivery: target P 0.282->0.494->0.698; top counts [2, 4, 4]/4
topic/refund: target P 0.184->0.285->0.400; top counts [0, 4, 4]/4
topic/technical: target P 0.155->0.273->0.426; top counts [0, 1, 4]/4
cancellation_intent/satisfied: expected level 1.615->1.592->1.562
cancellation_intent/frustrated: expected level 1.614->1.611->1.595
cancellation_intent/leaving: expected level 1.647->1.751->1.801
Limitations: exact duplicate-ID cue responded in the wrong direction; deadline sensitivity and numeric monotonicity remain weak; some security/needs_human responses remain overpositive. Do not claim these mechanisms are fixed.


Planned final forward25: paired hidden VALIDATION300 after restoring exact frozen best4 scalar.weight=w0+.875*d_mix900, candidate_weights/mix900_0875_scalar.pt. All other tensors original. SHA256 63b78a773a55a4725883275abfaf66b322dc96549d4b96953449e4d14116695d. Expected identical to forward19: .348000 -> .580667, gain .232667 SE .015184. Repeat is restoration/reproducibility check, not new independent evidence. Final submission will leave only this scalar.weight edit.

