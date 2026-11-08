# Decision-model weight repair
Initial state: no edits intended. Seed 82149, workflow-stratified fit/dev/test split {'fit': 448, 'dev': 224, 'test': 228}. Test IDs reserved for final paired measurement, not weight fitting.
Next forward: original DECIDE, all 900 practice IDs; establish baseline and inspect fit/dev errors. Exact edits: none.

Pass 1 baseline: original accuracy 0.565556, item SE 0.007137 over 900 examples. No edits. Fit/dev errors will drive diagnosis; test subset remains excluded from fitting.

Baseline fit/dev confusion summaries computed on 672 examples. Next pass: READ original, first 200 interleaved fit examples (1000 questions), IDs in read_fit1.json. Exact edits: none. Aim: locate scorer-accessible directions for observed option-ranking errors.

Diagnosis: fit/dev accuracies (672 examples): agent risk 39.9%, outcome 42.9%; invoice discrepancy severity 40.5%, urgency 42.3%; security severity 32.7%. Repeated confusions: invoice urgency gold elevated->routine 77/168, discrepancy gold material/severe->none 89/168; agent harmful->partial 33/168. High-confidence examples show explicit cues (overdue, prohibited freight, constraint violations) ignored or underused.
Pass 2 READ: 200 balanced fit examples, 1000 questions. Testing final scorer readout reconstruction, no backbone simulation.
Next pass: READ original second 200 balanced fit examples (1000 questions), exact IDs in read_fit2.json. Exact edits: none.

Pass 3: second 200 balanced fit examples read (1000 questions). Original scorer logits reconstruct to max error 3.8e-6 from head_inputs. Candidate mechanism: final readout of existing GELU features, not altered temperature or embeddings.
Next pass: READ original remaining 48 fit + first 152 dev examples (1000 questions), exact IDs in read_fitdev.json. Exact edits: none. Dev labels excluded from least-squares fitting.

Read alignment verified by reproducing recorded probabilities; from-filter retains baseline record order, not supplied ID order.
Next pass: READ original remaining 72 dev examples (360 questions), IDs in read_dev2.json. Exact edits: none. All 448 fit and 224 dev will have scorer activations; 228 test examples remain unseen for edit design.

All 448 fit + 224 dev scorer activations loaded and verified. Derivation: option-centered nonlinear scorer features, weighted equally per question; ridge solve for target temperature-scaled log gold residuals. Correction constrained perpendicular to original readout so not global scaling. Candidate floor/regularization screened on dev only; no weights saved yet.

Initial ridge (log gold floor .03, relative penalty .01) improves dev 55.00% -> 61.34%, paired SE 1.45pp, but delta norm 9.05 vs original readout norm .353. Exploring scorer spectral attenuation and direct normalized-input residual before choosing a causal intervention. No edits saved.

Input residual ridge (.01) dev 55.00% -> 64.82%, +9.82pp (selection set SE 1.66pp), Brier .05735 -> .03554. Final-GELU-only ridge peaks 62.14%. Hypothesis refined: fixed MLP loses/attenuates semantic option contrasts that remain in its normalized input. Will transmit a residual direction through an exact GELU +/- pair in two lowest-contribution neurons and test cue counterfactuals plus dose response before retaining.

Low-contribution scorer units identified from fit option-contrast RMS: neuron 33 (4.4e-8) and 746 (5.9e-8). Their near-zero original output weights permit an exact +/- GELU linear residual channel with negligible removal effect. No edits yet. Final closed-form target comparison includes soft/mode mixtures; test and validation remain unused.

Candidate frozen for causal tests: normalized-input residual, ridge relative=.01, target 5*T*(.75 gold probability + .25 gold mode), subtract original logits; fit on 448 examples. Delta norm 14.8355131149292. Exact implementation planned: +/- GELU pair using scorer neurons 33 and 746; original contrast contribution <1e-6. No edits yet.
Next pass: READ original 144 controlled requests (720 questions), exact requests/tags in counterfactual_original.json and counterfactual_tags.json. Preregistered effects: violations raise harmful/risk; errors reduce success; overdue/discount raises urgency; production privileges raise security severity.

Counterfactual source pass 6 read; predicted edit cue effects printed and saved in kernel. Current exact edits: head.scorer.1.weight rows 33/746 = +/- candidate_dx/||candidate_dx||; biases = -/+ projection of fit input mean; head.scorer.3.weight columns 33/746 = +/- 0.25*||candidate_dx||. All other values original. Definition candidate_quarter_definition.pt.
Next pass: DECIDE both on 224 dev IDs in dose_dev.json, strength .25. Preregistered local prediction: accuracy 59.821%, +4.821pp, paired SE 1.055pp.

Pass 7 actual quarter-strength intervention: original 55.000%, edited 59.821%, gain +4.821pp, paired SE 1.055pp; 58 improved / 21 worsened of 224 dev examples. Exactly matches local scorer prediction.

Next pass: DECIDE both on the same 224 dev IDs in dose_dev.json. Exact edits: same +/- rows/biases at neurons 33/746; final readout entries now +/- ||candidate_dx|| (strength 1.0), other values original. Definition candidate_full_definition.pt; copies candidate_linear_full/. Preregistered expected accuracy 67.054%, +12.054pp, paired SE 1.678pp.

Pass 8 actual full-strength intervention: original 55.000%, edited 67.054%, +12.054pp, paired SE 1.678pp; 119 improved / 39 worsened. Matches predicted scorer-only effect exactly, confirming causal edit location. Quarter->full dose gain grows as predicted.
Next pass: VALIDATION both on all 300 hidden examples. Exact edits remain candidate_linear_full: scorer.1 rows/biases 33/746 +/- residual direction and centering bias, scorer.3 entries +/- ||direction||, strength 1.0. No validation examples have been observed or used in fitting.

PASS 9 VALIDATION CONFIRMED best_1: 300 hidden examples, original 57.533%, edited 68.800%, +11.267pp, paired SE 1.520pp; 157 improved / 59 worsened. Backup edited directory to best_1/ and manifest. This is the current best confirmed edit.
Next pass: READ edited on the identical 144 controlled counterfactual requests in counterfactual_edited.json (720 questions). Exact edit: full-strength candidate_linear_full, same as validation. Preregistered increases: violation->harmful contrast .0945 -> .4163, risk .4106 -> 1.0435, overdue urgency .0429 -> .3792. Discount and freight not repaired by this channel.

Pass 10 edited counterfactual replay matches local predictions (max option-contrast error recorded above). Actual mean violation->harmful probability contrast .0945 -> .4163, risk expected-level contrast .4106 -> 1.0435, overdue urgency .0429 -> .3792. Discount and freight remain unrepaired. Investigating workflow-specific residual directions using fit data only; current submitted weights still validated best_1.

Workflow-level gated readouts had only small dev benefits and are not installed. Testing question-level interference via independent fit-only ridge residuals. Identity gates use workflow/question labels only (no gold answer labels); no current weight changes. Validated best_1 remains installed.

Exploratory mechanism: a shared linear residual cannot exploit state signal common to all option vectors; conditional q/option readouts allow different signs for that shared cue. Fit-only dual ridge solutions computed for fixed regularization grid; no edit installed, best_1 remains current. Gate fits use question/option identity only, not gold answers.

Next pass: DECIDE both same 224 dev IDs in dose_dev.json, shared channel strength 1.5, no conditional channels installed. Exact edits same scorer rows/biases33/746, output weights +/-1.5*||candidate_dx||. Preregistered prediction accuracy 62.679%, gain +7.679pp, paired SE 1.866pp; lower than full-strength 67.054%. Best_1 safely backed up.

Pass 11 over-strength 1.5x: dev 62.679%, +7.679pp SE1.866pp, exactly predicted and below 1x 67.054%. Confirms non-monotonic dose response; reverting to 1x for extension.

Next pass: DECIDE both 224 dev IDs in dose_dev.json. Exact current edits: full-strength base residual pair33/746 plus 54 option-gated residual channels using 108 low-contribution scorer neurons (indices in conditional_candidate.pt). Gates fit on q/option identities only; target questions selected by original fit accuracy <.60, residual ridge=.1; strength1, gate scale30, epsilon.025. Parameter copies conditional_candidate.pt. Local predicted dev result {'acc': 0.7008928656578064, 'gain': 0.15089285373687744, 'se': 0.01706094667315483, 'brier': 0.04147079959511757}.

Pass 12 actual conditional scorer: dev original55.000% ->70.089%, +15.089pp SE1.706pp, 135 improved/32 worsened. Exact local prediction confirmed. Increment over shared repair: 0.030357142857142857 SE 0.00967270631048533.
Next pass: VALIDATION both on300 hidden examples. Exact edits: conditional_candidate.pt (110 scorer neurons modified; base pair +54 option-gated directions on14 baseline-poor questions). No new fitting or changes since pass12.

PASS13 VALIDATION CONFIRMED best_2: original57.533% ->71.733%, +14.200pp SE1.543pp,171 improved/47 worsened. Best2 backup created; candidate conditional_candidate.pt. Conditional extension improves validation aggregate +2.933pp over best1 (no paired SE between hidden candidates available).
Next pass: READ edited identical144 counterfactual requests,720questions, in counterfactual_conditional.json. Exact current edit unchanged conditional_candidate.pt (110 neurons). Predicted effects: discount urgency contrast .0064->.3046, prohibited freight discrepancy .0507->.1887, overdue urgency .3792->.6032, violations harmful .4163->.5368. Compare with passes6 and10.

Pass14 actual conditional counterfactuals measured; local scorer replay remains numerically accurate. Confirmed state-dependent responses, including discount and freight cues absent from simpler repair. Exploring remaining invoice errors only on dev; no test or hidden-label access. Current best remains best_2 (validation +14.20pp SE1.543pp).


Next pass15: READ original 88 controlled invoice requests (440 questions) in invoice_cues_original.json. Sources ['train_726', 'train_605', 'train_654', 'train_881', 'train_674', 'train_788', 'train_666', 'train_854']; no probe labels fitted. Installed edits remain validated best_2: conditional_candidate.pt (scorer.1 rows/biases pair33/746 plus108 gated neurons; scorer.3 corresponding110 entries). Original model requested for cue baseline.


Pass15: controlled invoice cue baseline obtained (8 fit sources x11 variants,440questions). Current best2 unchanged. Testing numerical vs textual cue sensitivity using original scorer inputs; no backbone simulation or new fit labels.


Invoice cue subspace experiment: paired numeric/text cue directions fit with closed-form residual ridge on448 fit examples only; max dev increment0.089pp, not sufficient. Rejected; no additional cue-fit weights saved.
Next pass16: DECIDE both224 dev IDs in dose_dev.json. Exact edit conditional_half.pt: base linear pair33/746 unchanged at1x; same54 q/option gated directions on108 low-contribution neurons, their final output entries halved to +/-10 (conditional extension strength.5); other parameters unchanged from conditional_candidate.pt. Predicted {'acc': 0.6937500238418579, 'gain': 0.14374999701976776, 'se': 0.016961755231022835, 'brier': 0.03991265967488289}. best_2 remains backed up.


Pass16 actual half conditional extension: dev55.000% ->69.375%, gain14.375pp SE1.696pp,130 improved/33 worsened; exact local prediction.
Next pass17: DECIDE both224 dev IDs in dose_dev.json. Exact edit conditional_double.pt: same base pair33/746 at1x, same54 conditional channel input directions/gates/biases; conditional final readout entries +/-40 (2x), all other weights original or as best2. Expected {'acc': 0.6919642686843872, 'gain': 0.14196428656578064, 'se': 0.018559131771326065, 'brier': 0.048655249178409576}. best_2 backed up; diagnostic intervention only.


Pass17 actual conditional2x: dev55.000% ->69.196%, +14.196pp SE1.856pp,131 improved/39 worsened; matches local prediction, below full1x70.089%.
Next pass18: DECIDE both224 dev IDs in dose_dev.json. Exact diagnostic edit conditional_prior_only.pt: shared residual pair33/746 unchanged at1x;54 conditional gates retained, but each state direction removed and replaced by its448-fit mean correction for that q/option in the paired biases. Expected {'acc': 0.6669642925262451, 'gain': 0.11696428805589676, 'se': 0.017034942284226418, 'brier': 0.041881389915943146}. Best2 fully backed up.


Pass18 actual conditional state-direction ablation: dev55.000% ->66.696%, +11.696pp SE1.703pp. Full conditional repair exceeds prior-only by3.393pp, paired SE1.003pp on224dev items. Confirms conditional gain requires state-dependent directions, not only answer priors.
DESIGN FROZEN: restore best2 conditional_candidate.pt exactly; no further fitting/selection using test examples. Invoice-specific attempted extensions rejected for negligible dev benefit. All remaining interventions diagnostic only unless explicitly noted.
Next pass19: DECIDE both on228 untouched test practice IDs in frozen_test.json. Exact edit best2: scorer.1 rows/biases33/746 shared residual and108 low-contribution q/option gated rows; scorer.3 corresponding110 entries. Entirely identical to validation pass13. Test IDs were excluded from weight fitting and hyperparameter selection.


PASS19 INDEPENDENT TEST: frozen best2 original58.421% ->72.456%, gain14.035pp paired SE1.653pp,131 improved/37 worsened among228 reserved practice examples. No test fitting or selection will be done.
Next pass20: READ original first200 reserved test IDs in frozen_test_read.json (1000questions), frompass19original. Installed edit unchanged best2/conditional_candidate.pt. Purpose only gate-selectivity and displaced-neuron checks and frozen ablations; no weight fitting.


Pass20 frozen test read audit: first200 reserved items. Gate selectivity and original displaced-neuron contribution checked locally; no refitting. Best2 remains installed. Full test gain positive in all workflows: agent+24.21pp SE3.64; customer+9.47pp SE3.21; invoice+15.44pp SE3.17; security+7.02pp SE2.76.


Pass20 audit: reserved-test q/option gates TPR99.9646%, FPR0.00421%; displaced original-neuron option-contrast RMS0.000496. Local full scorer reproduces actual edited probabilities max error0.000428.
Next pass21: DECIDE both all228 reserved test IDs in frozen_test.json. Diagnostic shared-only ablation, exact shared_only_frozen.pt: scorer.1 rows/biases33/746 and scorer.3 entries33/746 as validated best1 atstrength1; other neurons original. No new fitting or dose selection. Compare with frozen best2 pass19 to measure independent contribution of conditional channels. Final choice remains validation-confirmed best2.


Pre-pass21 shared-only audit: raw logits include a centering constant; option-contrast error and dev metrics checked to ensure exact best1 behaviour. Exact installed shared_only_frozen.pt unchanged; next pass21 both228 reserved test IDs.


Pass21 shared-only reserved test:58.421% ->70.877%, +12.456pp paired SE1.596pp. Full best2 independent increment over shared=0.01578947368421053 SE=0.009653513042658705.
Next pass22: DECIDE both228 reserved test IDs in frozen_test.json. Exact diagnostic edit conditional_prior_only.pt: shared pair33/746 at1x,54 conditional gates retained but state-dependent vectors removed and replaced by per-q/option fit mean corrections. Identical ablation to pass18, no test refitting. Final submitted choice remains best2, the largest validation-confirmed gain.


Pass22 prior-only reserved test:58.421% ->70.351%, +11.930pp SE1.618pp. Full-minus-prior-only=0.02105263157894737 SE=0.009847499527089491. No refitting.
Restored frozen best2 exactly (conditional_candidate.pt). Next pass23: READ original200 requests/1000questions in metadata_robustness_original.json: first25 reserved examples per workflow, each unchanged and with irrelevant record_reference added. Selection by pre-existing split order, not outcomes. Installed edits are best2, but original requested. Preregistered expectation: substantive answers and the repair's gain should be largely stable. Pass24 will replay edited; pass25 final validation.


Pass23 metadata robustness baseline read obtained. Original={'original_state_accuracy': 0.593999981880188, 'metadata_added_accuracy': 0.5759999752044678, 'answer_flip_rate': 0.13600000739097595, 'mean_probability_total_variation': 0.07536449283361435}; frozen best2 local prediction={'original_state_accuracy': 0.7279999852180481, 'metadata_added_accuracy': 0.6859999895095825, 'answer_flip_rate': 0.18799999356269836, 'mean_probability_total_variation': 0.13291752338409424}.
Next pass24: READ edited identical200 requests/1000questions in metadata_robustness_edited.json. Exact installed edit conditional_candidate.pt/best2:110 scorer neurons modified (shared pair33/746 and54 conditional paired channels), no changes since restoration beforepass23. No fitting or selection; check actual robustness and scorer replay.


Pass24 actual metadata perturbation replay: {'original_state_accuracy': 0.7279999852180481, 'metadata_added_accuracy': 0.6859999895095825, 'answer_flip_rate': 0.18799999356269836, 'mean_probability_total_variation': 0.13291071355342865}; paired gains {'unchanged': {'gain': 0.1340000030398369, 'paired_se': 0.024627323817256814}, 'metadata_added': {'gain': 0.11000000223517419, 'paired_se': 0.025800321536657934}}. Repair remains ahead, but invariance is imperfect and worse than original by flip-rate metric. No tuning based on this diagnostic.
FINAL AUDIT: exact best2 restored and checked tensor-equal to conditional_candidate.pt, only head.scorer.1.weight/.bias and head.scorer.3.weight edited;112860 entries total. Backup final_best/; hashes and measurements final_manifest.json.
Next pass25 FINAL VALIDATION both300 hidden examples. Exact edit unchanged best2 (shared residual neurons33/746 plus54 gated residual pairs in108 low-contribution neurons); same as pass13. No further fitting or changes; expected deterministic repeat of57.533% ->71.733%, +14.200pp.

