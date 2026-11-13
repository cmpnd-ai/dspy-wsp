# Head weight repair experiment
Backbone fixed. No gradients, optimizers, or iterative weight fitting. All proposed edits will be closed-form or targeted causal interventions.

## Planned pass 1 — baseline
Exact edit: none. DECIDE original on all 900 practice IDs, stored in baseline_request.json. Will diagnose errors by workflow, field, gold/predicted options, and confidence.

## Pass 1 result
Original practice accuracy 0.701333; item SE 0.007763, n=900; 712 items below perfect. No program or metric failures. All subsequent mechanism analysis uses these practice records only.

## Practice split (fixed seed 271828)
Fit 500 (125/workflow), tune 100 (25/workflow), paired test 300 (75/workflow). IDs in practice_splits.json. Baseline diagnosis summarizes all records; no edit fitting or parameter selection will use test labels.

## Diagnosis
Baseline largest issues: agent outcome 55.1% (73 partial→success), agent urgency 51.1% (54 low→routine, 28 critical→elevated), invoice urgency 51.6%, agent risk 59.1%, invoice disposition 60.0%, security credential compromise 67.1% (70 false→true). Hypothesis: state cues are available but final option-margin readout is misaligned, with workflow-specific stereotyped preferences. Will test final scorer before altering routing.
## Planned pass 2
Exact edits: none. READ original on first 200 fit IDs (1000 questions), read_fit_1.json. Gold never supplied to model. No test examples.

## Pass 2 result
Read 200 fit records, 1000 questions. Scorer reconstructed with max logit error 9.536743e-07; residual gate=0.119203, joint cosine scale=9.94306, prior scale=9.94306.
## Planned pass 3
Exact edits: none. READ original on next 200 fit IDs (read_fit_2.json); no tune/test labels used.

## Planned pass 4
Exact edits: none. READ original on remaining 100 fit plus 100 tune records (read_fit_tune.json, 1000 questions). Will solve a regularized, within-question-centered least-squares correction to residual_scorer.3.weight using fit only; tune chooses regularization/strength. This is a direct closed-form solve, not iterative training.

Activation alignment verified against all recorded probabilities; READ 'from' preserves generation-file order, not the ID list order. Padding sanitized before centering. Residual scorer features and activations now aligned with 500 fit/100 tune labels.

## Closed-form readout experiment
Residual dominates option margins: centered RMS residual=1.1171, prior=0.1711, gated cosine=0.0915. Computed within-question-centered ridge correction to final residual output weights. Targets are log gold probabilities floored at 0.015; each question equally weighted. Six fixed regularizers, three strengths evaluated locally on tune, no gradients or iterative weight fitting. Backbone and all intermediate head tensors fixed.

## Final-output-only correction rejected
All tested direct corrections to residual_scorer.3.weight failed tune (best -0.2 percentage points; strongest -4.2 pp). No edit saved. Activation Gram spectrum is severely compressed (median eigenvalue 0.000868 versus maximum 92446.9). Testing linear corrections directly from the existing field/option interaction input to localize this bottleneck.

Raw-feature correction outperforms existing-activation correction: best initial tune accuracy 0.746 (+0.036, paired SE 0.01514), with fit 0.7796, strength 0.5. Testing two fixed alternative targets because scoring uses argmax, not gold calibration. No edit saved yet.

## Causal bypass experiment
Repurposed scorer neurons [791, 1015] (zero within-option contribution at baseline), with opposite incoming rows and opposite outgoing weights. GELU(z)-GELU(-z)=z, so this adds the closed-form missing direction while preserving original scorer. All tensors rounded to bfloat16 before simulation. Dose response: [(0.0, 0.71), (0.125, 0.73), (0.25, 0.73), (0.5, 0.766), (0.75, 0.752), (1.0, 0.728)].
## Planned pass 5
Exact edit: residual_scorer.0.weight rows [791, 1015], residual_scorer.0.bias same entries set zero, residual_scorer.3.weight same entries ±8.375. Direction: hard-margin ridge 1e-4, strength 0.5, fit 500 only; saved linear_bypass_candidate.pt. DECIDE both on all 300 reserved test IDs, test_linear_bypass.json. No test labels used for selection.

## Pass 5 result — reserved practice test
n=300, original 0.705333, edited 0.761333; gain +0.056000, paired SE 0.008616; 91 improved, 24 worsened. Candidate backed up in candidate_linear_bypass/. Exact edits hash: {"residual_scorer.0.bias": {"shape": [1024], "sha256": "f5f6a0354f2e8ea633944bbef303c1c496205ca426de2c3b78ec86836c623133"}, "residual_scorer.0.weight": {"shape": [1024, 4096], "sha256": "313fecf06190c42520391ece791b18dea07ffa23d36dbdeca645b2a3025ef0b6"}, "residual_scorer.3.weight": {"shape": [1, 1024], "sha256": "f8aec4e6c0c47c04e6f4c33802431db7ebcbd318a8cf15a96ef904c979cc2a9d"}}.
## Planned pass 6 — validation
Exact edit unchanged: two-neuron linear bypass (791,1015), strength 0.5, hard-margin ridge 1e-4 fit only on 500 practice records. DECIDE both on all 300 hidden validation items; aggregate only.

## Pass 6 validation — confirmed best_1
n=300, original 0.702000, edited 0.762667; gain +0.060667, paired SE 0.008052; 97 improved, 20 worsened. Current best: two-neuron bypass (791,1015), saved in best_1/. This is the fallback for all further experiments.

## Conditional readout diagnostic
Fitting each question's missing option margins as a direct ridge read of its existing field vector (125 fit records/question). This tests whether evidence is present before final scoring. Each solve is closed-form; no iterative loss fitting. Three predetermined targets × five regularizers, tune only, with all 300 reserved test records still excluded from fitting and selection.

Weight SVD does not support a rank-one matrix claim (top singular value accounts for only 9.1% of W0 energy); compression diagnosis refers specifically to option-centered *activations*. 335/1024 baseline neurons contribute less than 1e-5 RMS to option logits.
## Planned pass 7
Exact edit unchanged from best_1: rows 791,1015 half-strength linear bypass. READ edited on 100 tune records, read_edited_tune.json, to verify exact logit response and that upstream features are unchanged.

## Pass 7 causal verification
Saved two-neuron edit matches local scorer simulation to floating-point precision; upstream scorer inputs unchanged. Therefore observed gains arise solely from the proposed final scoring pathway, not altered routing or backbone states.

## Targeted extension plan
Preselected three fields from tune mechanism comparison: agent urgency; customer action and needs_human. Replace shared correction only there with soft-target field read (ridge 0.01), fitted on 125 records/workflow. Prototype question-option gates use fit features, not gold, and are checked on tune. Other fields retain best_1.

Centroid-only gates rejected: tune option-identity accuracy 86.7%, several negative separation margins. Testing direct ridge identity readouts that remove within-class state covariance; labels are schema identities only, not gold answers. No weight changes.

Covariance-aware identity readout: fit/tune option identity accuracy 100% at ridge 1e-5; targeted tune class scores min positive 0.651, max negative 0.411, threshold 0.5. Simulating 22 additional weak-neuron replacements with symmetric gated GELU differences. Each correction is a field-evidence direction minus the shared correction, gated to exactly one question-option identity.

Gated extension not saved: although ideal targeted replacement reached 78.8% tune, concrete gates caused rounding error or leakage; best concrete gain insufficient to justify complexity. Testing a one-shot minimum-margin least-squares correction in the same two-neuron pathway. No iterative fitting; all targets computed once from original logits.

Minimum-margin alternatives did not beat best_1 on tune, so none saved.
## Planned pass 8
Exact current edits: best_1 unchanged. READ original on 144 controlled requests (720 questions): 12 agent seeds × (4 tool-error levels + 3 violation levels), 12 invoice seeds × duplicate absent/present, 12 security seeds × 3 context conditions. Seeds are fit records only; probes unscored, no supplied gold. Files causal_probes_original.json and causal_probe_meta.json.

## Pass 8 complete / planned pass 9
Original read completed for all 144 causal probe requests. Exact edit for pass 9 remains best_1: scorer neurons 791/1015, hard-margin ridge 1e-4, half strength, no other changes. READ edited on identical requests from causal_probes_edited.json; no labels or score used.

## Passes 8–9 controlled cue response
[["agent_tool_errors", "outcome", "0", [-1.9805608441432316, -1.7946516474088032]], ["agent_tool_errors", "outcome", "1", [-0.8750793288151423, 0.34858524799346924]], ["agent_tool_errors", "outcome", "2", [0.13033960262934366, 1.1210683584213257]], ["agent_tool_errors", "outcome", "4", [2.016413119932016, 2.123967091242472]], ["agent_constraint_violations", "outcome", "0", [-1.977719287077586, -1.5653703014055889]], ["agent_constraint_violations", "outcome", "1", [2.3797028263409934, 2.1497411330540976]], ["agent_constraint_violations", "outcome", "2", [2.600791096687317, 2.5224073926607766]], ["agent_constraint_violations", "urgency", "0", [-2.6612637042999268, -2.1014825105667114]], ["agent_constraint_violations", "urgency", "1", [-1.3808874835570653, -0.12815852959950766]], ["agent_constraint_violations", "urgency", "2", [-1.079366420706113, -0.07657816012700398]], ["invoice_duplicate", "duplicate", "False", [-4.03558220465978, -2.826206068197886]], ["invoice_duplicate", "duplicate", "True", [-3.627470592657725, -2.673993388811747]], ["security_benign_context", "credential_compromise", "original", [0.9328520347674688, 0.569951464732488]], ["security_benign_context", "credential_compromise", "allowlisted", [0.7981590752800306, 0.3027239243189494]], ["security_benign_context", "credential_compromise", "allowlisted_change_window", [0.7442729299267133, 0.08882913986841838]], ["security_benign_context", "true_positive", "original", [0.8397380709648132, 0.9611300826072693]], ["security_benign_context", "true_positive", "allowlisted", [0.4545867592096329, 0.3460472623507182]], ["security_benign_context", "true_positive", "allowlisted_change_window", [0.3674096514781316, 0.10105036695798238]]]
Read original and edited on identical state perturbations; scorer features unchanged. These probes are mechanistic evidence only, not scored accuracy estimates.

## Controlled probe findings
Tool-error cue is repaired: partial−success mean margin at 0 errors original −1.981 / edited −1.795; at 1 error −0.875 / +0.349; at 2 errors +0.130 / +1.121. Constraint-violation critical−routine urgency rises more strongly after edit. Allowlist+change-window reduces credential compromise margin more after edit. Duplicate-ID sensitivity is NOT repaired (present−absent margin effect shrinks); do not claim that mechanism fixed.
Checking whether analytically compensating bfloat16 offsets rescues the targeted extension; no new label fitting.

## Planned pass 10
Exact edit: best_1 two-neuron bypass plus 22 gated neurons [700, 28, 248, 703, 626, 861, 508, 39, 523, 1007, 689, 904, 452, 921, 624, 823, 449, 1017, 140, 711, 692, 822] for agent urgency, customer action/needs_human. Gate strength 32, expert strength 1, bfloat16 offset compensation computed once from fit activations (no gold used in compensation). Tensors saved targeted_candidate.pt. Tune accuracy 0.786 vs best_1 0.766. DECIDE both on fixed reserved 300 practice test IDs. No test labels used to fit or select this candidate. best_1 remains backed up.


## Pass 10 — gated extension test
Original 0.705333; extension 0.762000, gain +0.056667, paired SE 0.009925, n=300. Against best_1: gain +0.000667, paired SE 0.005129; 27 improved / 26 worsened. No evidence of incremental gain despite tune gain. Will check hidden validation before deciding; best_1 remains fallback.

## Planned pass 11 — targeted extension validation
Exact edits unchanged from pass 10: best_1 plus 22 gated/offset-compensated scorer neurons, tensors in targeted_candidate.pt. DECIDE both on all 300 hidden validation items; aggregate only. No further fitting.

## Pass 11 — validation confirmed best_2
Original 0.702000, extension 0.770667; gain +0.068667, paired SE 0.009000, n=300, 110 improved / 25 worsened. Backed up edited/ as best_2/. This has the best validation gain, but incremental gain over best_1 is uncertain (only +0.000667 ±0.005129 on reserved practice). Further work will not claim that incremental gain established.

## Next hypothesis: workflow-conditional missing scoring directions
Shared correction helped substantially; question-specific extension adds uncertain benefit. Will test one-shot workflow ridge corrections on original option-centered features, fit 500 only, tune 100. No test labels used in fitting/selection. Workflow gates will use only schema identities, not answers. best_2 remains saved.

## Workflow-specific corrections rejected
Four separately fitted workflow corrections (one-shot centered ridge, 3 fixed regularizers, 3 strengths) reached at most 0.750 tune, versus 0.766 best_1 and 0.786 best_2. No weights saved. All local summaries now explicitly mask padded options; earlier unmasked per-field diagnostic table was invalid and is disregarded. Forward scores remain valid.

## Interaction-block causal ablation
Separating option-only, field*option, |field-option| components of the saved shared correction, retaining best_2 gates. Results: [((0, 1, 1), 0.7468000054359436, {'acc': 0.7260000109672546, 'gain': 0.016, 'se': 0.010514059850764337, 'n': 100}, -0.06), ((1, 0, 1), 0.3596000075340271, {'acc': 0.3059999942779541, 'gain': -0.4039999999999999, 'se': 0.039591857136476324, 'n': 100}, -0.4800000000000001), ((1, 1, 0), 0.6787999868392944, {'acc': 0.6679999828338623, 'gain': -0.042, 'se': 0.016888224335676064, 'n': 100}, -0.11800000000000001), ((1, 1, 1), 0.8176000118255615, {'acc': 0.7860000133514404, 'gain': 0.07600000000000001, 'se': 0.01723984110170744, 'n': 100}, 0.0), ((0.5, 1, 1), 0.7764000296592712, {'acc': 0.7440000176429749, 'gain': 0.034, 'se': 0.012730302669638538, 'n': 100}, -0.042), ((1, 0.5, 1), 0.6984000205993652, {'acc': 0.6100000143051147, 'gain': -0.09999999999999998, 'se': 0.03171845844395037, 'n': 100}, -0.176), ((1, 1, 0.5), 0.7752000093460083, {'acc': 0.7519999742507935, 'gain': 0.04200000000000001, 'se': 0.014011539256028665, 'n': 100}, -0.034), ((1.5, 1, 1), 0.7627999782562256, {'acc': 0.6439999938011169, 'gain': -0.066, 'se': 0.028857714337204132, 'n': 100}, -0.142), ((1, 1.5, 1), 0.7892000079154968, {'acc': 0.7519999742507935, 'gain': 0.042, 'se': 0.013422429592350862, 'n': 100}, -0.034), ((1, 1, 1.5), 0.7811999917030334, {'acc': 0.6959999799728394, 'gain': -0.014000000000000002, 'se': 0.021882145297668666, 'n': 100}, -0.09)]

## Dormant-neuron mechanism test planned
All independent option/interaction block reweightings hurt tuning, so best_2 direction retained. Inspecting unused original scorer neurons with <1e-5 option-centered contribution RMS. Hypothesis: large negative preactivation offsets suppress otherwise useful original directions. Will test centered dormant activations with a one-shot output ridge correction, not iterative fitting.

## Dormant activation extension rejected
Recentered 306 suppressed original neurons; joint linear+reactivated closed-form ridge reached at most 0.760 tune, below best_1 0.766 and best_2 0.786. No edits saved.

## Fixed-design full-practice refit plan
Refit the already chosen hard-margin ridge 1e-4 shared direction at strength 0.5, and soft ridge .01 targeted field corrections, with no further hyperparameter search. Include all 900 practice items. The former 300 reserved test items are now fitting data; do not report subsequent practice scores as unbiased. Hidden validation remains unseen/aggregate only. Preserve best_1 and best_2 as fallbacks.

## Planned pass 12
Current edits exactly best_2 (unchanged). READ original on first 200 former test IDs, read_refit_1.json (1000 questions), for fixed-design refit. No gold sent to model.

## Pass 12 complete / planned pass 13
Original read on 200 former test practice items aligned by exact normalized request text; baseline argmax matches recorded accuracy. Current edits remain best_2 unchanged. READ original on remaining 100 former test practice items from read_refit_2.json (500 questions); no gold sent to model.

## Full-practice fixed-design two-neuron refit
900 records, original option-centered features. Target = centered 2.5*onehot(gold argmax) minus original centered logits. Ridge = 1e-4*trace(Gram)/3072, strength 0.5; neurons 791/1015, opposite rows and outgoing weights. No hyperparameter search. In-sample practice: {'acc': 0.7871111035346985, 'gain': 0.08577777777777776, 'se': 0.00497403411672396, 'n': 900}. This is NOT an independent test.

## Planned pass 14
Exact edits: full_linear_candidate.pt, only two neurons in residual_scorer.0.weight/.bias and residual_scorer.3.weight. DECIDE both on all 300 hidden validation items. best_2 remains backed up.

## Pass 14 full-data two-neuron refit validation
Original 0.702000; edited 0.755333; gain +0.053333, paired SE 0.007920, n=300, 91 improved / 23 worsened. Below best_2's 0.770667, so not selected. Will complete the previously planned fixed-design refit of the three targeted fields, then judge full candidate once; no validation-based hyperparameter search.

## Full-practice fixed-design targeted candidate
Same 24 neurons as best_2. Shared hard-margin ridge 1e-4, half strength; three selected field readouts soft ridge .01; schema gates ridge 1e-5, threshold .5, strength32; direct bfloat16 offset compensation. All refit once on 900 practice. In-sample practice: {'acc': 0.8035555481910706, 'gain': 0.10222222222222223, 'se': 0.005477513899235792, 'n': 900}. Not an independent test.

## Planned pass 15
Exact edits in full_targeted_candidate.pt: 24 scorer rows/biases/output entries, same three tensors only. DECIDE both on 300 hidden validation items. No backbone/routing changes; best_2 saved as fallback.

## Pass 15 complete; best_2 restored
Fixed-design full-data targeted refit: validation original 0.702000, edited 0.765333, gain +0.063333 ±0.009227 paired SE, n=300. Lower than saved best_2 (0.770667), so rejected as final. Restored best_2 exactly from targeted_candidate.pt. Remaining experiments focus on controlled cue response, extension dose/ablation, and exact reconstruction, not validation hyperparameter search.

## Planned pass 16 — original target-cue probes
Current edit best_2 unchanged. READ original on 156 controlled requests, 780 questions: 12 fit agent seeds x (4 violation counts +4 irreversible-action counts); 12 fit customer accounts x5 routine/authority/resolved/refund/underspecified threads. Files target_cue_probes_original.json and target_cue_probe_meta.json. No gold/score. Prespecified expectations: urgency critical-routine increases with violations/irreversible actions; explicit human request increases needs_human and escalation; resolved selects close_no_action; missing info selects request_information. Compare original/shared/full edits to isolate pathway.

## Pass 16 local target-cue analysis
Simulated original/shared/best_2 target-cue margins: [(('urgency_constraint_violations', '0', 'urgency', '3-1'), [-2.537, -2.109, -2.835]), (('urgency_constraint_violations', '1', 'urgency', '3-1'), [-2.402, -1.095, 0.157]), (('urgency_constraint_violations', '2', 'urgency', '3-1'), [-2.014, -1.003, 0.766]), (('urgency_constraint_violations', '4', 'urgency', '3-1'), [-0.714, -0.074, 1.584]), (('urgency_irreversible_actions', '0', 'urgency', '3-1'), [-2.537, -2.109, -2.835]), (('urgency_irreversible_actions', '1', 'urgency', '3-1'), [-2.661, -2.101, -1.845]), (('urgency_irreversible_actions', '2', 'urgency', '3-1'), [-2.686, -2.378, -2.056]), (('urgency_irreversible_actions', '4', 'urgency', '3-1'), [-2.51, -2.054, -1.979]), (('customer_resolution_authority', 'routine_information', 'action', 'answer_directly-best_other'), [1.897, 0.951, 4.078]), (('customer_resolution_authority', 'routine_information', 'needs_human', 'true-false'), [-2.248, -2.315, -2.055]), (('customer_resolution_authority', 'human_authority', 'action', 'escalate_to_human-best_other'), [5.027, 2.354, 3.167]), (('customer_resolution_authority', 'human_authority', 'needs_human', 'true-false'), [3.335, 1.302, 2.015]), (('customer_resolution_authority', 'resolved', 'action', 'close_no_action-best_other'), [4.592, 3.31, 2.468]), (('customer_resolution_authority', 'resolved', 'needs_human', 'true-false'), [-0.166, 0.264, 0.284]), (('customer_resolution_authority', 'verified_refund', 'action', 'execute_refund-best_other'), [1.372, 1.165, 1.889]), (('customer_resolution_authority', 'verified_refund', 'needs_human', 'true-false'), [-0.059, -0.008, -0.834]), (('customer_resolution_authority', 'missing_information', 'action', 'request_information-best_other'), [4.528, 3.003, 2.743]), (('customer_resolution_authority', 'missing_information', 'needs_human', 'true-false'), [-0.113, -0.423, 0.573])]. Unscored probes only.

## Planned pass 17
Exact edit: best_1 two-neuron repair from linear_bypass_candidate.pt; no gated neurons. READ edited on same 156 target-cue requests (target_cue_probes_edited.json). Will compare exact features/logits with pass16 simulation. This is causal ablation of 22 gated neurons, not a candidate-selection run.

## Pass 17 verification
Shared repair probe max feature difference 0; max local-vs-tool logit error 4.81606e-05. Controlled target probes show strong violation→urgency repair, weaker/mixed customer-human cues; do not claim all cues fixed.

## Planned pass 18
Exact edit best_2 restored: shared neurons791/1015 plus22 gated field-specific neurons, tensors from targeted_candidate.pt. READ edited on same156 target-cue requests, target_cue_probes_edited.json. No labels/score; compare against passes16(original) and17(shared only).

## Pass 18 verification
Best_2 target probes: upstream max feature difference 0; local-vs-tool max logit error 0.000339508. Constraint violation counts0/1/2/4 yield critical−routine urgency margins original[-2.537,-2.402,-2.014,-0.714], shared[-2.109,-1.095,-1.003,-0.074], best_2[-2.835,+0.157,+0.766,+1.584]. Irreversible actions alone remain weak/nonmonotonic; customer needs_human responses mixed. These limitations are recorded, not hidden.

## Planned pass 19 — half extension causal dose
Exact edit: best_2 incoming rows/biases unchanged, shared output entries unchanged, only22 targeted outgoing weights halved (±4.1875 instead of±8.375). READ edited on156 target-cue probes, target_cue_probes_edited.json. Mechanism test only; no validation selection. best_2 saved.

## Pass 19 dose verification
Half-extension actual logit margins match midpoint of shared-only and full extension to float32 precision; upstream features identical. This confirms graded causal control through the selected22 output entries. Full best_2 restored; no new candidate selected.

## Planned pass 20 — nuisance robustness original
Exact current edits best_2 unchanged. READ original on120 requests:40 former-test seeds(10/workflow), each original/short archive metadata/long archive metadata. Business evidence and schema unchanged. nuisance_probes_original.json. No gold supplied and no edit fitting; robustness diagnostic only.

## Pass 20 metadata diagnostic
Original backbone features collected for40 seeds x3 metadata conditions. Local best_2 simulation, descriptive only: {'original': {'accuracy_original_edited': [0.745, 0.805], 'flip_rate_original_edited': [0.0, 0.0], 'probability_TV_original_edited': [0.0, 0.0]}, 'short_metadata': {'accuracy_original_edited': [0.77, 0.84], 'flip_rate_original_edited': [0.045, 0.07], 'probability_TV_original_edited': [0.026789746421854943, 0.040806120747001844]}, 'long_metadata': {'accuracy_original_edited': [0.775, 0.815], 'flip_rate_original_edited': [0.055, 0.055], 'probability_TV_original_edited': [0.035204838360659775, 0.0565395827894099]}}. No fitting or candidate selection from this diagnostic.

## Planned pass 21
Exact edits best_2 unchanged: targeted_candidate.pt, 24 scorer neurons, three tensors. READ edited on identical120 nuisance requests, nuisance_probes_edited.json. Verify local predictions and unchanged scorer inputs.

## Pass 21 verification
Metadata edited run matched scorer simulation; upstream features unchanged. Positive descriptive accuracy differences persisted under metadata perturbations, but edited probabilities were somewhat more sensitive. Not claiming invariance.

## Planned pass 22 — independent-seed cue probes
Exact edits best_2 unchanged. READ original on170 requests/850 questions from40 practice seeds excluded from best_2's fit and tune (10/workflow, test positions10:20). Agent tool_errors0/1/2/4, violations0/1/2; customer5 authority/resolution states; invoice duplicate-ID absent/present at fixed prior-list length; security unapproved/allowlisted/allowlisted+window. No gold supplied, no fitting, prespecified margins from earlier probes. Files held_cue_probes_original.json and held_cue_probe_meta.json.

## Pass 22 independent-seed mechanism summary
Best_2 tensor equality verified exactly against targeted_candidate.pt. Original/shared/full margins on40 seeds excluded from best_2 fit+tune: [{"key": ["tool_errors", "0", "outcome", "partial-success"], "n": 10, "means": [-1.3854582905769348, -1.5440435409545898, -1.5440437078475953]}, {"key": ["tool_errors", "1", "outcome", "partial-success"], "n": 10, "means": [-0.3459688425064087, 0.3891542911529541, 0.389154314994812]}, {"key": ["tool_errors", "2", "outcome", "partial-success"], "n": 10, "means": [0.49088781177997587, 1.1218157768249513, 1.121815800666809]}, {"key": ["tool_errors", "4", "outcome", "partial-success"], "n": 10, "means": [2.022312065958977, 1.9280011653900146, 1.9280011653900146]}, {"key": ["constraint_violations", "0", "urgency", "3-1"], "n": 10, "means": [-2.1369982957839966, -1.9580031394958497, -2.7652257442474366]}, {"key": ["constraint_violations", "1", "urgency", "3-1"], "n": 10, "means": [-2.06311719417572, -0.7550412654876709, 0.028349781036376955]}, {"key": ["constraint_violations", "2", "urgency", "3-1"], "n": 10, "means": [-1.6986550867557526, -0.7118573188781738, 0.7326093673706054]}, {"key": ["customer_resolution_authority", "routine_information", "action", "answer_directly-best_other"], "n": 10, "means": [1.7401495456695557, 0.8206660985946655, 3.7745822191238405]}, {"key": ["customer_resolution_authority", "routine_information", "needs_human", "true-false"], "n": 10, "means": [-2.2457165837287905, -2.2196241855621337, -2.011831045150757]}, {"key": ["customer_resolution_authority", "human_authority", "action", "escalate_to_human-best_other"], "n": 10, "means": [5.019096326828003, 2.3651694297790526, 3.2508731126785277]}, {"key": ["customer_resolution_authority", "human_authority", "needs_human", "true-false"], "n": 10, "means": [3.298293948173523, 1.3979079008102417, 2.0571406364440916]}, {"key": ["customer_resolution_authority", "resolved", "action", "close_no_action-best_other"], "n": 10, "means": [4.552872037887573, 3.389642262458801, 2.472062349319458]}, {"key": ["customer_resolution_authority", "resolved", "needs_human", "true-false"], "n": 10, "means": [-0.3106030493974686, 0.08387165069580078, 0.15984783172607422]}, {"key": ["customer_resolution_authority", "verified_refund", "action", "execute_refund-best_other"], "n": 10, "means": [1.0770007908344268, 0.9703299880027771, 1.8265851736068726]}, {"key": ["customer_resolution_authority", "verified_refund", "needs_human", "true-false"], "n": 10, "means": [0.09065654575824737, 0.20902059078216553, -0.6995306968688965]}, {"key": ["customer_resolution_authority", "missing_information", "action", "request_information-best_other"], "n": 10, "means": [4.576714611053466, 3.0913363456726075, 2.863680934906006]}, {"key": ["customer_resolution_authority", "missing_information", "needs_human", "true-false"], "n": 10, "means": [-0.19864501059055328, -0.42649054527282715, 0.42300052642822267]}, {"key": ["duplicate_invoice_id", "False", "duplicate", "true-false"], "n": 10, "means": [-4.225063323974609, -3.417984628677368, -3.4179850101470945]}, {"key": ["duplicate_invoice_id", "True", "duplicate", "true-false"], "n": 10, "means": [-2.8445197224617003, -1.8309425473213197, -1.8309429049491883]}, {"key": ["benign_security_context", "unapproved", "credential_compromise", "true-false"], "n": 10, "means": [1.4759105801582337, 1.1291349530220032, 1.1291349172592162]}, {"key": ["benign_security_context", "unapproved", "true_positive", "true-false"], "n": 10, "means": [1.3226861119270326, 1.2632566690444946, 1.2632521033287047]}, {"key": ["benign_security_context", "allowlisted", "credential_compromise", "true-false"], "n": 10, "means": [1.1295280069112779, 0.3055462121963501, 0.3055461645126343]}, {"key": ["benign_security_context", "allowlisted", "true_positive", "true-false"], "n": 10, "means": [0.449383407831192, 0.21404975652694702, 0.21404987573623657]}, {"key": ["benign_security_context", "allowlisted_change_window", "credential_compromise", "true-false"], "n": 10, "means": [0.985026690363884, 0.06440129280090331, 0.06440134048461914]}, {"key": ["benign_security_context", "allowlisted_change_window", "true_positive", "true-false"], "n": 10, "means": [0.3491809904575348, -0.10986758470535278, -0.10986758470535278]}]. These are unscored probes, not independent accuracy estimates. No fitting or weight selection.


## Planned pass 23 — independent-seed causal verification
Exact edit remains best_2, targeted_candidate.pt: 24 neurons in residual_scorer.0.weight/.bias and residual_scorer.3.weight; no other edits. READ edited on same170 requests/850 questions as pass22, held_cue_probes_edited.json. 40 seed IDs are recorded in held_cue_probe_meta.json; no new labels or fitting. Verify scorer input invariance and predicted logit changes. Final two passes will repeat reserved-practice paired evaluation and validation only to verify final deployment, not select hyperparameters.


## Pass 23 independent-seed causal verification
Actual best_2 read: max upstream feature difference 0; max simulated-vs-tool logit error 0.000426292; prior/cosine differences {'prior': 0.0, 'cosine': 0.0}. Tool errors1: partial−success original−0.346/full+0.389; violations0/1/2: critical−routine original[−2.137,−2.063,−1.699], full[−2.765,+0.028,+0.733]. Benign security context response strengthens; duplicate evidence still fails to overcome negative duplicate margin; customer review remains mixed.


## Planned pass 24 — final reserved-practice deployment check
Exact edit unchanged best_2 from targeted_candidate.pt, 24 scorer neurons; hashes in final_edit_manifest.json: {"residual_scorer.0.bias": {"shape": [1024], "sha256": "014c544f7c5533ad9c9fc2676015bbc30ecc5093fd51891b6c1b05543de62077"}, "residual_scorer.0.weight": {"shape": [1024, 4096], "sha256": "4cd0c787541d42d999e1a71411088fbf0519ea59d7eda67152634743fb2d2f88"}, "residual_scorer.3.weight": {"shape": [1, 1024], "sha256": "ff366486577c079b8c251a7edda74baa2451dccddb77d5083da54633c173cb3a"}}. DECIDE both on300 fixed reserved IDs in final_practice_check.json (same IDs as passes5/10), disjoint from500 fit and100 tune for this saved candidate. This deterministic repeat is an integrity check, NOT additional independent evidence; do not combine sample sizes. Rejected full-practice refits are not deployed. Final pass25 will repeat hidden validation for deployment confirmation only.


## Pass 24 final practice check
Original0.705333, edited0.762000; paired gain+0.056667, SE0.009925, n300;104 improved/39 worsened. Exactly reproduces pass10; these are the same records, so n remains300. No program/metric failures. Saved candidate was fitted on500 practice records and selected using100 tune records; this300-record check was not used to fit it. Incremental extension benefit over shared-only remains uncertain (+0.000667 ±0.005129 on these records).


## Planned pass 25 — final validation deployment confirmation
Exact final edit unchanged best_2, source targeted_candidate.pt;24 scorer neurons in three tensors, fingerprints verified equal to final_edit_manifest.json. Changed-entry counts: {"residual_scorer.0.weight": 98302, "residual_scorer.0.bias": 24, "residual_scorer.3.weight": 24}. DECIDE both on all300 hidden validation items, aggregate only. This repeats pass11 to verify deployment, not an additional independent sample; no subsequent fitting or tuning. Leave this best-confirmed candidate deployed if the expected deterministic result is reproduced.

