# Weight repair experiments
No edits yet. All analyses use practice data only; validation will be used only for aggregate confirmation.

## Pass 1 planned: baseline
Exact edits: none. Original model on all 900 practice IDs in baseline_request.json. Fixed split (splits.json): 400 discovery, 450 confirmation, 50 reserve. Inspect individual failures only on discovery; confirmation records not used to derive directions.

Baseline pass 1: accuracy 0.56556, item SE 0.00714 on 900. Discovery question breakdown computed; no edits. Confirmation records not inspected.

## Pass 2 planned: discovery activations A
Exact edits: none. Read original on 200 discovery cases (50/workflow, first half of discovery; read_a.json), all 1000 questions, layers 14 and 16. Hypothesis: systematic benign/routine/level-2 defaults reflect a shared option-scoring readout mismatch. Discovery accuracies: security severity .33; agent risk .40; invoice discrepancy .40; invoice urgency .41; customer action .44. Need inspect returned activations before selecting a component.

Read pass 2 ordering corrected by matching complete requests; corrected mappings are request_ids2. head_inputs are pre-scorer-LayerNorm, NOT post-GELU. No network computations simulated. Contrasts use centered unit-length exported vectors as geometric analysis.

## Pass 3 planned: discovery activations B
Exact edits: none. Read remaining 200 discovery cases (50/workflow; read_b.json), all 1000 questions. Purpose: test replication of invoice deadline double-difference and invoice-specific option contrast on independent discovery half. Pooled error direction rejected as too close to generic anti-confidence. No edits computed from confirmation.

Pass 3 read complete. Examining replication of invoice deadline direction and weight-only singular modes. No edits yet; candidate mechanisms not accepted without causal interventions.

## Pass 4 planned: causal head-mode attenuation, strength .25
Exact edit: head.head.layers.0.linear2.weight = original - .25 * sum(top two SVD modes), no other changes. [{"name": "head.head.layers.0.linear2.weight", "delta_norm": 2.4653589725494385, "relative_delta": 0.10719961673021317, "sha256": "d18f545f8d3c2d2cf5f3712f08eaa766e95a4bfe29b6fd9787f495cf47a4d5ab"}]
Read edited model on the same 200 discovery-A cases (mode_read.json), 1000 questions. Predict reduction of fixed routine/benign dominance and relative increase of state signal. This is exploratory; no confirmation examples used.

Pass 4 result: .25 attenuation of first-head-MLP top two modes gave exactly zero accuracy/ranking gain on discovery A (.558 unchanged). Logit correlation .999872; best scalar .97811; non-scalar residual 1.60%. Invoice urgency margin near/later unchanged (-2.826/-2.932). This strength is rejected as effectively a confidence rescaling.

## Pass 5 planned: head-mode attenuation dose response, strength .85
Exact edit: head.head.layers.0.linear2.weight = original - .85 * sum(top two SVD modes); other weights original. [{"name": "head.head.layers.0.linear2.weight", "delta_norm": 8.38223934173584, "relative_delta": 0.36447951197624207, "sha256": "abb2d3b8f11aeb67e2d320852d83969a8c76e279efe70208361e341be957cc5c"}]
Read edited on discovery A, same 200 cases/1000 questions in mode_read.json. Compare to strengths 0 and .25; retain only if non-global useful ranking changes occur.

Pass 5 analyzed; head MLP attenuation experiment finished and restored to original. Moving to targeted scorer input-direction interventions. Deadline double-difference replicated on B: near +.0242 versus later -.0214 in centered-unit head features; corresponding original elevated-routine raw logit gaps remain -2.789/-2.947.

Pass 5: .85 mode attenuation discovery A +.0060 +/- .00280 paired SE, but invoice urgency unchanged; logit corr .9907, scale .8194. Restored; not kept.
## Pass 6 planned: matched payment-deadline counterfactuals
Exact edits: none. Original reads 64 synthetic requests (16 discovery-A invoices x later/near/overdue/discount), all 320 questions. Only payment days/status/discount fields change; deadline_counterfactuals.json and tags record exact cases. Goal: isolate causal deadline representation before targeted scorer edit.

## Pass 7 planned: causal payment readout, gain 2
Exact edit: scorer neurons 594,746 (original output weights -3.41e-6,+3.75e-6) receive +/- protected payment direction divided coordinatewise by scorer LN gain; biases cancel LN offset; output weights +/-2. GELU antisymmetry creates a linear directional readout. Direction derived from 16 matched discovery-A invoice counterfactuals, projected off 53 fixed option/nuisance means and 32 unrelated-contrast PCs from discovery A. Other tensors original. [{"name": "head.scorer.1.bias", "delta_norm": 0.03503570333123207, "relative_delta": 0.05556487664580345, "sha256": "8ea608ae977ae641fbfe296e97b3332c1a5a1613723cb6f7ba9865dbfae8cb38"}, {"name": "head.scorer.1.weight", "delta_norm": 1.630679965019226, "relative_delta": 0.08426794409751892, "sha256": "30c39c7f0edcb85ece7b503b84adcbb7fbf0e8fd0677bc6f45938b5639920adf"}, {"name": "head.scorer.3.weight", "delta_norm": 2.828432083129883, "relative_delta": 8.017189025878906, "sha256": "de3a719c17f99db42259edb662dafe4e44316646404088f16a330dae7eeaa4be"}]
Read discovery B (200 cases/1000 questions, payment_read_b.json), compare original pass 3. Predict increased urgency for near/overdue invoices with limited unrelated changes. This is discovery/tuning, not final confirmation.

Pass 7 results analyzed below in session; gain-2 checkpoint saved to payment_gain2. Upstream feature equality checked.

Pass 7 payment gain2: discovery B accuracy .566 vs .557, paired +.0090 SE .00669 (26 improved,16 worse). Invoice urgency .38->.52; near-due elevated-routine mean -2.789->-.600, later -2.947->-2.462. Head features bit-identical. Invoice disposition -.06 and customer action +.08 reveal collateral effects. Not yet sufficient for retention.

## Pass 8 planned: payment-direction dose response, gain 4
Exact same two-neuron direction as pass7, output coefficients now +/-4. [{"name": "head.scorer.1.bias", "delta_norm": 0.03503570333123207, "relative_delta": 0.05556487664580345, "sha256": "8ea608ae977ae641fbfe296e97b3332c1a5a1613723cb6f7ba9865dbfae8cb38"}, {"name": "head.scorer.1.weight", "delta_norm": 1.630679965019226, "relative_delta": 0.08426794409751892, "sha256": "30c39c7f0edcb85ece7b503b84adcbb7fbf0e8fd0677bc6f45938b5639920adf"}, {"name": "head.scorer.3.weight", "delta_norm": 5.656859397888184, "relative_delta": 16.034364700317383, "sha256": "f5892f7562a0422dcb885c877a16f56a174f9a216247b7c70401f7e01c5c0ac3"}]
Read same 200 discovery-B cases/1000 questions via payment_read_b.json. Compare gains 0,2,4 for causal dose-response; confirmation remains unused.

Pass 8 payment gain4: discovery B .574 vs .557, paired +.0170 SE .00871 (41 improved,24 worse). Invoice urgency .38->.62; near-due elevated-routine margin +1.588 versus -.600 at gain2 and -2.789 original; later margin -1.978 and all 24 still routine. Checkpoint payment_gain4. Gain not yet independently confirmed.

## Pass 9 planned: matched agent counterfactuals
Current edits remain payment gain4, but READ ORIGINAL ignores them. Original reads 16 discovery-A agent records x 8 controlled counter/autonomy variants =128 requests/640 questions (agent_counterfactuals.json). All other fields fixed per seed. Discovery finding: all 20 constraint-violation traces gold harmful, original partial; risk high only 10/20. Goal: isolate state cue directions for a targeted counter readout, not encode label frequency.

## Pass 10 planned: add violation readout, gain 2
Exact edits: payment cue gain4 in neurons594/746 plus refined violation cue gain2 in neurons518/33. Refined direction uses violation1/3 minus irreversible-only matched traces; nulls clean/dry/tool-error/irreversible target means, all other agent question means across variants, other workflow means and 32 unrelated contrast PCs. Direction residual norm .0443; discovery-B harmful-partial projection .0299 for violations versus .0026 without. [{"name": "head.scorer.1.bias", "delta_norm": 0.039755020290613174, "relative_delta": 0.06304948031902313, "sha256": "4cfbf3f37a85c6537bce5a38da53069b6d2011c71f8837aa4ef8443266d77b22"}, {"name": "head.scorer.1.weight", "delta_norm": 2.3214051723480225, "relative_delta": 0.11996226012706757, "sha256": "f45585e4b7174096c7e3ae73aad5687d6824f0ee11ab7a9004d605cfaa9aaa3c"}, {"name": "head.scorer.3.weight", "delta_norm": 6.324559688568115, "relative_delta": 17.926959991455078, "sha256": "765d6e14cf8aff4bb61a605c8280513b6481b5b23cc370bb1d04cf35189c27ac"}]
Read 200 discovery-B cases/1000 questions via payment_read_b.json. Predict selective improvement on agent outcome/risk/urgency. Compare both original pass3 and payment-only pass8.

Pass 10 payment4+violation2: discovery B .588 vs .557, paired +.0310 SE .01013 (47 improved,23 worse). Violated traces n10: outcome 0/10 original ->9/10 edited; risk 6/10->10/10; urgency4/10->7/10. Nonviolated outcome26/40->24/40, risk14/40->16/40. Checkpoint payment4_violation2.

## Pass 11 planned: violation dose response, gain4
Payment cue remains gain4; violation cue gains now +/-4 (same directions and rows). [{"name": "head.scorer.1.bias", "delta_norm": 0.039755020290613174, "relative_delta": 0.06304948031902313, "sha256": "4cfbf3f37a85c6537bce5a38da53069b6d2011c71f8837aa4ef8443266d77b22"}, {"name": "head.scorer.1.weight", "delta_norm": 2.3214051723480225, "relative_delta": 0.11996226012706757, "sha256": "f45585e4b7174096c7e3ae73aad5687d6824f0ee11ab7a9004d605cfaa9aaa3c"}, {"name": "head.scorer.3.weight", "delta_norm": 8.00000286102295, "relative_delta": 22.676002502441406, "sha256": "403c2dbf64d5a4ef1e8e485fb7c5cf31073cb6a5c89c8a5c6f6cf58a399c3641"}]
Read same 200 discovery-B records/1000 questions, payment_read_b.json. Compare violation gains0/2/4; look for saturation and collateral effects.


Pass 11 result: payment4+violation4 read completed; comparing discovery-B dose response against gains0/2. Checkpoint payment4_violation4.


Pass11 payment4+violation4: discovery B .585 vs .557, paired +.0280 SE .01099. Increment over violation2 -.0030 SE .00539. Violated outcome10/10, risk10/10, urgency8/10; collateral errors increase. Prefer violation2. Features compare only at valid options (padding may contain NaNs).


## Pass 12 planned: independent practice confirmation
Exact edit: restored payment4_violation2 (payment +/-4 rows594/746, violation +/-2 rows518/33), all other tensors original. [{"name": "head.scorer.1.bias", "delta_norm": 0.039755020290613174, "relative_delta": 0.06304948031902313, "sha256": "4cfbf3f37a85c6537bce5a38da53069b6d2011c71f8837aa4ef8443266d77b22"}, {"name": "head.scorer.1.weight", "delta_norm": 2.3214051723480225, "relative_delta": 0.11996226012706757, "sha256": "f45585e4b7174096c7e3ae73aad5687d6824f0ee11ab7a9004d605cfaa9aaa3c"}, {"name": "head.scorer.3.weight", "delta_norm": 6.324559688568115, "relative_delta": 17.926959991455078, "sha256": "765d6e14cf8aff4bb61a605c8280513b6481b5b23cc370bb1d04cf35189c27ac"}]
DECIDE both on 450 predesignated confirmation IDs in confirm_initial.json; none used to derive/select directions. Valid exported head features are bit-identical; previous equality failure was NaN padding.


Pass12 independent confirmation (450 cases): original .564889, edited .585333, paired gain +.020444 SE .007220; 103 improved,64 worsened. No confirmation records inspected to construct edits.


## Pass 13 planned: hidden validation of initial repair
Exact edits unchanged: payment4_violation2, [{"name": "head.scorer.1.bias", "delta_norm": 0.039755020290613174, "relative_delta": 0.06304948031902313, "sha256": "4cfbf3f37a85c6537bce5a38da53069b6d2011c71f8837aa4ef8443266d77b22"}, {"name": "head.scorer.1.weight", "delta_norm": 2.3214051723480225, "relative_delta": 0.11996226012706757, "sha256": "f45585e4b7174096c7e3ae73aad5687d6824f0ee11ab7a9004d605cfaa9aaa3c"}, {"name": "head.scorer.3.weight", "delta_norm": 6.324559688568115, "relative_delta": 17.926959991455078, "sha256": "765d6e14cf8aff4bb61a605c8280513b6481b5b23cc370bb1d04cf35189c27ac"}]
DECIDE both on all 300 hidden validation cases; aggregate confirmation only. No validation examples accessible.


Pass13 hidden validation, initial repair: original .575333, edited .591333, paired +.016000 SE .008449;69 improved,45 worsened. Positive but modest-power confirmation; best validated checkpoint best_1.


## Pass 14 planned: causal security impact cues
Current edits unchanged best_1 payment4_violation2, but READ ORIGINAL. Original reads 20 discovery-A security seeds x8 asset-criticality/privilege variants, 160 requests/800 questions (security_counterfactuals.json). Asset low/medium/high/crown_jewel crossed with repo:read vs deploy:production,secrets:read; alert evidence otherwise held fixed. Test whether severity-contrast representations track these explicit state cues while raw logits default to medium.


## Pass 15 planned: security impact readout, low dose
Exact edit: payment4+violation2 plus crown2,production2,highasset1 on scorer pairs [(993, 924), (166, 732), (381, 365)]. Directions from matched discovery-A counterfactual contrasts, projected off 194-212 nuisance means/PCs. Residual norms crown .03004, production .01264, highasset .01788. [{"name": "head.scorer.1.bias", "delta_norm": 0.06409169733524323, "relative_delta": 0.10164623707532883, "sha256": "79499dd4774f2403dc9947fc88f108c43e0656e2d8c983e8cbd944db23ba9bf8"}, {"name": "head.scorer.1.weight", "delta_norm": 3.740475654602051, "relative_delta": 0.193294957280159, "sha256": "ae98c0a1730c5f02b42410b463b940f7c66733bc0f4d3557f17d3ef556c43bbf"}, {"name": "head.scorer.3.weight", "delta_norm": 7.615800857543945, "relative_delta": 21.58698272705078, "sha256": "0e5fbbee4a809929556ccef98940539166e381da4468701bd7bb16790d858bbf"}]
Read 200 discovery-B cases/1000 questions via payment_read_b.json. Predict selective security severity changes. Baseline for additional edit is pass10. No confirmation records used for directions.


Pass15 security low (crown2/prod2/high1): discovery B accuracy .596, gain vs original +.0390 SE .01019; increment vs base repair +.0080 SE .00692. Security severity .38->.50. Production-access 3-2 margin -.790->+.092; unprivileged -.634->-.679. Some customer/action collateral (-.06) and invoice/disposition improvement (+.06). Checkpoint security_low.


## Pass 16 planned: security readout dose response, high
Exact edit: payment4+violation2 unchanged, security crown4/prod4/highasset2, same six rows/directions as pass15. [{"name": "head.scorer.1.bias", "delta_norm": 0.06409169733524323, "relative_delta": 0.10164623707532883, "sha256": "79499dd4774f2403dc9947fc88f108c43e0656e2d8c983e8cbd944db23ba9bf8"}, {"name": "head.scorer.1.weight", "delta_norm": 3.740475654602051, "relative_delta": 0.193294957280159, "sha256": "ae98c0a1730c5f02b42410b463b940f7c66733bc0f4d3557f17d3ef556c43bbf"}, {"name": "head.scorer.3.weight", "delta_norm": 10.58304214477539, "relative_delta": 29.99762535095215, "sha256": "8a55109179dc26d217f31e4931610634cb6383a137bf9d4c70267ff975192fab"}]
Read 200 discovery-B cases/1000 questions via payment_read_b.json. Compare security strengths0/low/high; no confirmation data used to select strength.


Pass16 high security dose: discovery B .581 vs original .557, +.0240 SE .01191; increment vs base repair -.0070 SE .00923. Severity .46 versus .50 low dose. Production margin increases monotonically -.790,+.092,+.973, but high dose creates collateral errors. Reject high; refine specificity using discovery-A nuisance geometry only.


## Pass 17 planned: narrative-protected security cues, moderate dose
Exact edit: base payment4+violation2 plus security crown3,production3,highasset2,lowimpact6 on pairs [(993, 924), (166, 732), (381, 365), (624, 477)]. Cue directions from same matched discovery-A counterfactuals, remove 128 unrelated-question PCs plus 48 within-condition narrative PCs and fixed nuisance means. Low-impact direction is low unprivileged minus medium unprivileged, contrast low-vs-medium, protected against privileged/high/crown means. [{"name": "head.scorer.1.bias", "delta_norm": 0.07639878988265991, "relative_delta": 0.12116467207670212, "sha256": "e785175e6afcabe438ab2dc29cd232ec8d7d1282f37b542354d043d70875b5df"}, {"name": "head.scorer.1.weight", "delta_norm": 4.094645023345947, "relative_delta": 0.21159720420837402, "sha256": "d0afb4307f15a0fb1582246cbff35b9d43a10edfe4010e0959eba3ced2a3422a"}, {"name": "head.scorer.3.weight", "delta_norm": 12.490035057067871, "relative_delta": 35.40299606323242, "sha256": "526781dd39a04a7397f76b1292c5afe6f3b1b5f34dfab32a71d7426fa1fc5675"}]
Read discovery B 200 cases/1000 questions via payment_read_b.json. Compare actual model output; no network simulation. Expect lower contamination of criticality readout by alert wording.


Pass17 refined security moderate: discovery B .577, gain +.0200 SE .01236 vs original, -.0110 SE .00932 vs validated base. Severity .52 but substantial unrelated errors; reject as global linear add-on. Next test localized nonlinear gate using question-identifying feature direction.


## Pass 18 planned: gate-localized security readouts, moderate dose
Exact edit: base payment4+violation2 plus same four security cue directions/gains3,3,2,6 as pass17. Security pairs now rows (12*question_gate +/- cue)/scorerLN_gain, biases cancel LN offset then subtract9.5, outputs +/-gain/2. Gate derived from discovery-A question means protected against option/content/cue variation. B gate projections security severity [.0352,.0635], others[-.0213,.0146], clear separation. [{"name": "head.scorer.1.bias", "delta_norm": 26.959999084472656, "relative_delta": 42.7572135925293, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "delta_norm": 34.80899429321289, "relative_delta": 1.7988094091415405, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "delta_norm": 8.306654930114746, "relative_delta": 23.545207977294922, "sha256": "5a31b4a219e9dcef3ba346d060185267126b68ea1e2670d0dc0b376c0345ee33"}]
Read discovery B 200 cases/1000 questions via payment_read_b.json. Predict preserved severity effects but disappearance of unrelated changes versus validated base repair pass10. Gating has not yet been accepted causally.


Pass18 gated security moderate: discovery B .595 vs original .557, +.0380 SE .01016; increment vs base repair +.0070 SE .00329 (9 improved,2 worsened). Security severity .38->.52. Gate preserves severity centered logit RMS .75687 (ungated .75686) while off-target RMS .41233->.00003; all 950 unrelated question rankings unchanged. Causal localization confirmed. Checkpoint security_gated_moderate.


## Pass 19 planned: gated security dose response, strong
Exact edit: validated base payment4+violation2 plus gated security crown6,production6,highasset4,lowimpact12, same gate/directions as pass18. [{"name": "head.scorer.1.bias", "delta_norm": 26.959999084472656, "relative_delta": 42.7572135925293, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "delta_norm": 34.80899429321289, "relative_delta": 1.7988094091415405, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "delta_norm": 12.490035057067871, "relative_delta": 35.40299606323242, "sha256": "526781dd39a04a7397f76b1292c5afe6f3b1b5f34dfab32a71d7426fa1fc5675"}]
Read discovery B 200 cases/1000 questions via payment_read_b.json. Compare gains zero/moderate/strong; predict selective severity change with no unrelated rankings affected.


Pass19 gated strong: discovery B .594 vs original .557, +.0370 SE .01071; increment vs base +.0060 SE .00446. Severity .50 versus moderate .52; severity probability squared error .19376 versus moderate .14215/base .16790. No unrelated ranking changes. Prefer moderate gated security (3,3,2,6) based on discovery only.


## Pass 20 planned: independent practice confirmation of gated security extension
Exact edit restored security_gated_moderate: payment4+violation2 plus gated security crown3/production3/highasset2/lowimpact6. [{"name": "head.scorer.1.bias", "delta_norm": 26.959999084472656, "relative_delta": 42.7572135925293, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "delta_norm": 34.80899429321289, "relative_delta": 1.7988094091415405, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "delta_norm": 8.306654930114746, "relative_delta": 23.545207977294922, "sha256": "5a31b4a219e9dcef3ba346d060185267126b68ea1e2670d0dc0b376c0345ee33"}]
DECIDE both on fixed 450 confirmation IDs, confirm_security.json. No case records or activations from confirmation used to derive directions, gate or doses; earlier base-repair aggregate was seen.


Pass20 confirmation summary: {"n": 450, "before": 0.564888888888889, "after": 0.5955555555555555, "gain": 0.03066666666666667, "paired_se": 0.007477360038162653, "improved": 119, "worsened": 59}
Increment versus validated base repair (same cases): {"n": 450, "before": 0.5853333333333333, "after": 0.5955555555555555, "gain": 0.010222222222222223, "paired_se": 0.0026631173277569926, "improved": 30, "worsened": 7}
Only aggregate scores examined; no confirmation examples used to construct or tune edits.

## Pass 21 planned: hidden validation of gated security repair
Exact edit: payment4+violation2 plus gated security crown3/production3/highasset2/lowimpact6; unchanged from passes18/20. [{"name": "head.scorer.1.bias", "changed": 12, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "changed": 12288, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "changed": 12, "sha256": "5a31b4a219e9dcef3ba346d060185267126b68ea1e2670d0dc0b376c0345ee33"}]
DECIDE both on all 300 hidden validation cases, aggregate results only. No tuning since independent confirmation.

Pass21 hidden validation: original .5753333333, edited .598, paired gain +.0226666667 SE .0085033056;75 improved,43 worsened. Best validated checkpoint best_2 (identical to security_gated_moderate).
Remaining passes may test localization of existing payment/violation directions; best_2 retained unless a new candidate passes independent confirmation and validation.

## Pass 22 planned: causal workflow gate for violation readout
Exact edits: best_2 unchanged except violation pair518/33 now (32*agent_gate +/- violation_direction)/LN_gain, bias cancels LN offset and subtracts 21.489385605; output weights +/-1 preserve effective gain2 when active. Agent gate from discovery-A workflow means, nulling option/question means, other workflows, 64 within-question PCs and violation direction. [{"name": "head.scorer.1.bias", "delta_norm": 40.60368347167969, "relative_delta": 64.39541625976562, "sha256": "ddc43b55de88d400e62ebe90a8313fecbda1bb76e1d52b7a376dede915fe915e"}, {"name": "head.scorer.1.weight", "delta_norm": 57.73789596557617, "relative_delta": 2.983696460723877, "sha256": "a810a901d9e2b6e1d2fbbdabe643f647845edcdb5e828bc91170fc4b9ce00e8b"}, {"name": "head.scorer.3.weight", "delta_norm": 7.937286376953125, "relative_delta": 22.498233795166016, "sha256": "8151193a865ce2647f2e75f90742fd8b3fe7fd0c8f6cfcc159a7c0e5493e3e2c"}]
READ edited on the same 200 discovery-B cases/1000 questions via payment_read_b.json. Prediction: agent logits nearly unchanged versus pass18, unrelated violation leakage removed. Gate scale/threshold chosen from discovery-A separation, no confirmation fitting.

Pass22 agent-workflow gate: discovery B .593 vs original .557 (+.0360 SE .010101); versus best_2 -.0020 SE .004481. Agent predictions unchanged, but removing off-workflow effects does not improve accuracy. Reject candidate; retain validated best_2. Remaining three passes reserved for matched-cue audit, untouched reserve cases and final discovery evaluation.

## Pass 23 planned: final matched-cue audit
Exact edit restored best_2 (validated pass21), [{"name": "head.scorer.1.bias", "delta_norm": 26.959999084472656, "relative_delta": 42.7572135925293, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "delta_norm": 34.80899429321289, "relative_delta": 1.7988094091415405, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "delta_norm": 8.306654930114746, "relative_delta": 23.545207977294922, "sha256": "5a31b4a219e9dcef3ba346d060185267126b68ea1e2670d0dc0b376c0345ee33"}]
READ edited 192 counterfactual requests/960 questions: all64 payment requests from pass6, first64 agent requests from pass9, first64 security requests from pass14. Exact inputs and mapping in final_cue_audit.json/final_cue_audit_map.json. Compare with original cached reads, no metric fitting or strength selection. Prediction: stronger selective responses to deadline, violation and asset/privilege cues.

Pass23 final matched-cue audit: max absolute head-input difference 0.0; exact feature preservation confirms scorer-only intervention.
(6, 'later', 'elevated-routine'): n=16, original=-2.6413, edited=-2.7374
(6, 'near', 'elevated-routine'): n=16, original=-1.6891, edited=+6.6884
(6, 'overdue', 'elevated-routine'): n=16, original=-2.9478, edited=+1.9125
(6, 'discount', 'elevated-routine'): n=16, original=-2.7364, edited=-1.8891
(9, 'clean', 'harmful-partial'): n=8, original=-3.5437, edited=-3.4570
(9, 'clean', 'high-medium'): n=8, original=+0.9447, edited=+0.8128
(9, 'clean', 'elevated-routine'): n=8, original=-2.1940, edited=-2.8166
(9, 'dry', 'harmful-partial'): n=8, original=-3.8017, edited=-3.7690
(9, 'dry', 'high-medium'): n=8, original=+0.7603, edited=+0.4526
(9, 'dry', 'elevated-routine'): n=8, original=-2.7398, edited=-3.3995
(9, 'error1', 'harmful-partial'): n=8, original=-3.5646, edited=-3.5966
(9, 'error1', 'high-medium'): n=8, original=+0.9602, edited=+0.8540
(9, 'error1', 'elevated-routine'): n=8, original=-1.7956, edited=-2.1329
(9, 'error4', 'harmful-partial'): n=8, original=-3.4078, edited=-3.5688
(9, 'error4', 'high-medium'): n=8, original=+1.0633, edited=+0.9247
(9, 'error4', 'elevated-routine'): n=8, original=-1.5162, edited=-1.8142
(9, 'irreversible', 'harmful-partial'): n=8, original=-2.8919, edited=-2.8671
(9, 'irreversible', 'high-medium'): n=8, original=+0.0867, edited=-0.0543
(9, 'irreversible', 'elevated-routine'): n=8, original=-0.6097, edited=-0.9900
(9, 'violation1', 'harmful-partial'): n=8, original=-1.1816, edited=+1.1315
(9, 'violation1', 'high-medium'): n=8, original=+0.1645, edited=+1.6753
(9, 'violation1', 'elevated-routine'): n=8, original=-0.1472, edited=+0.7064
(9, 'violation3', 'harmful-partial'): n=8, original=-1.2063, edited=+0.8915
(9, 'violation3', 'high-medium'): n=8, original=+0.3940, edited=+1.8798
(9, 'violation3', 'elevated-routine'): n=8, original=-0.0510, edited=+1.0243
(9, 'uncontrolled', 'harmful-partial'): n=8, original=-1.0586, edited=+1.1100
(9, 'uncontrolled', 'high-medium'): n=8, original=+0.0959, edited=+1.0658
(9, 'uncontrolled', 'elevated-routine'): n=8, original=+0.4385, edited=+1.2745
(14, ('low', False), 'low-medium'): n=8, original=-1.5934, edited=-0.2387
(14, ('low', True), 'low-medium'): n=8, original=-1.8427, edited=-1.8991
(14, ('medium', False), 'high-medium'): n=8, original=-0.3898, edited=-0.2998
(14, ('medium', True), 'high-medium'): n=8, original=-0.4929, edited=+0.5845
(14, ('high', False), 'high-medium'): n=8, original=+0.0826, edited=+0.7938
(14, ('high', True), 'high-medium'): n=8, original=-0.1931, edited=+1.5374
(14, ('crown_jewel', False), 'critical-medium'): n=8, original=-1.4700, edited=+1.0942
(14, ('crown_jewel', True), 'critical-medium'): n=8, original=-1.3113, edited=+1.7587

## Pass 24 planned: untouched reserve
Exact edit best_2, unchanged since pass21: [{"name": "head.scorer.1.bias", "delta_norm": 26.959999084472656, "relative_delta": 42.7572135925293, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "delta_norm": 34.80899429321289, "relative_delta": 1.7988094091415405, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "delta_norm": 8.306654930114746, "relative_delta": 23.545207977294922, "sha256": "5a31b4a219e9dcef3ba346d060185267126b68ea1e2670d0dc0b376c0345ee33"}]
DECIDE both on 50 predesignated reserve IDs in final_reserve.json; not previously used to design or select any edit. No further weight changes planned.

Pass24 untouched reserve (50): {"n": 50, "before": 0.636, "after": 0.616, "gain": -0.02, "paired_se": 0.02231499907401901, "improved": 7, "worsened": 15}
Pooled independent practice confirmation+reserve (500): {"n": 500, "before": 0.572, "after": 0.5976, "gain": 0.025599999999999998, "paired_se": 0.0071155141122876235, "improved": 126, "worsened": 74}
Reserve regression is retained in reporting; no edits made in response. Best remains best_2: hidden validation gain +.022667 SE .008503 and pooled independent practice gain +.0256.

## Pass 25 planned: final discovery-set audit
Exact edit unchanged best_2, [{"name": "head.scorer.1.bias", "delta_norm": 26.959999084472656, "relative_delta": 42.7572135925293, "sha256": "ace43f55971dff4c7cb63641c38deab390d5a09271ff85350ce0d17d0b0d1844"}, {"name": "head.scorer.1.weight", "delta_norm": 34.80899429321289, "relative_delta": 1.7988094091415405, "sha256": "180975c53d243831aadd7d6fa17cb85797f976942fc39adc6eeeda22ec7f2ae9"}, {"name": "head.scorer.3.weight", "delta_norm": 8.306654930114746, "relative_delta": 23.545207977294922, "sha256": "5a31b4a219e9dcef3ba346d060185267126b68ea1e2670d0dc0b376c0345ee33"}]
DECIDE both on all400 discovery IDs in final_discovery.json. These are the development examples, so their gains are in-sample and will not be used as independent confirmation. No tuning or further changes planned; completion provides a complete 900-practice audit.
