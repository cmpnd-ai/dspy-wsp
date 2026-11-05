Initial diagnosis. No edits. Next: original baseline on all 900 practice examples.

Baseline 900: accuracy .51593 SE .01233. Fixed split: 140 per workflow fit, 140 per workflow untouched practice test, 20 per workflow reserve. Next read original head features for 420 fit cases (980 questions). No edits.
Read pass 2: reconstructed scorer logits max error 2.86e-6. Fit order is original baseline order; padded NaNs removed. Testing analytic residual ridge correction of final scorer readout; no edits yet.
Next pass: read 420 untouched practice test cases, original weights.

Candidate A: only head.scorer.3.weight = original + ridge residual delta, lambda=1, strength=1; fitted 420 cases. Untouched practice replay 420: .51984 -> .69365, gain .17381 SE .0221. Half strength .52778; 1.5 strength .38016. Next paired actual program on 420 practice test IDs. All other tensors original.
Pass 4 actual paired practice confirms exactly: .51984 -> .69365, +.17381 SE .02206, 183 improved/68 worsened. Next validation candidate A, unchanged head.scorer.3.weight; no other edits.
Pass 5 validation candidate A confirmed: .54333 -> .68222, gain .13889 SE .02627; 130 improved, 51 worsened. Saved best_A and candidate_readout.pt. Further work uses practice only for selection.
Candidate B: final scorer readout direct centered gold ridge lambda=.1, fit same 420. Practice replay test .72063 vs baseline .51984 (+.20079 SE .0238); sensors .8429 vs A .7357. Next paired practice test run, B only final readout edited.
Pass 6 confirms candidate B practice .72063, gain .20079 SE .02377, 194 improved/73 worsened. Next validation B unchanged.
Pass 7 B validation .69000 vs .54333, gain .14667 SE .02833. Saved best_B. Next causal tests separate fixed-option feature subspace from record-sensitive complement.
Next pass original read of 60 reserve cases; current edit B (read original ignores edit).
Candidate B reserve replay 0.61111, baseline 0.52778. Candidate C same direct ridge .1 fitted pooled 840 explored cases, excludes 60 reserve; reserve replay 0.63333. Next paired actual 60 reserve with C, final scorer only.
Pass 9 C reserve actual .52778 -> .63333, gain .10556 SE .06162, too small a sample for confidence. Next validation C, unchanged final readout.
Pass 10 C validation .73000 vs .54333, gain .18667 SE .02685; 134 improved/41 worsened. Best confirmed edit C saved best_C/candidate_C.pt. Further passes are causal diagnostics, not validation tuning.
Next paired test: C half-dose, w=w0+.5*(wC-w0), 420 explored practice cases. Best C safely saved.
Pass 11 C half-dose actual .54524 vs .51984, gain .02540 SE .00613. Next 90% C dose on 420 explored test cases.
Pass 12 C 90% dose .71270, gain .19286 SE .01908 on explored test. Next full C on same 420 (included in C fit; causal, not generalization).
Pass 13 full C explored-practice .80635, +.28651 SE .02262 (fit-included). Dose sequence .5=.54524, .9=.71270, 1=.80635. Next overshoot 1.1 same cases; predicted .69603.
Pass 14 overshoot 1.1 .69603, confirms nonmonotonic dose-response. Next C bias-only ablation: project delta onto rank-13 mean centered question/option subspace. 420 explored practice cases.
Pass 15 bias-only .41667, gain -.10317 SE .01707. Next evidence-only complement, same 420 explored practice cases; predicted .54603.

Pass 16 evidence-only complement .54603 vs .51984, gain .02619 SE .00607; 32 improved/5 worsened. Full C .80635 on same explored cases; both subspaces are needed. Current edit is diagnostic ablation, NOT final.

Restored full C, head.scorer.3.weight only. Next pass 17: paired 420 fitids (C-fit included), consistency diagnostic only; no selection on validation.

Pass 17 full C fit partition .51032 -> .80238, gain .29206 SE .02131 (fit-included).

Next pass 18: only top 64 scorer-coordinate corrections (ranked by centered feature energy times delta squared); paired 420 testids, fit-included. Diagnostic, not a replacement candidate.

Pass 18 top64 .47619, gain -.04365 SE .01041. Next pass 19 top256 corrections only, same 420 testids; predicted .58810.

Pass 19 top256 .58810, gain .06825 SE .01550. Next pass 20 top512 corrections only, same 420 testids; predicted .67778.

Pass 20 top512 .67778, gain .15794 SE .02011. Restored full C. Next pass 21 READ edited on 420 fitids (980 questions), to verify unchanged head features and predicted logits. Only head.scorer.3.weight edited.

Pass 21 feature verification: max head-input difference 0.0; scorer replay error 0.03134489059448242. Next pass 22 reverse direction w0-(wC-w0), same 420 testids, diagnostic negative control.

Pass 22 reversed direction .50952, gain -.01032 SE .00425. Restored C. Unchanged scorer output bias 0.031294167041778564; bias-corrected replay max error 5.054473876953125e-05. Next pass 23 edited READ 420 testids to verify second partition features. Final candidate remains C.

Pass 23 C second-partition edited read completed; upstream features checked against original. Next pass 24 READ edited 60 reserve cases (excluded from C fit). Full C only final scorer readout remains installed. Then final unchanged-C validation confirmation.

Pass 24 reserve head verification complete. Final edit: head.scorer.3.weight only, full candidate C; all other tensors untouched. Closed-form ridge readout, lambda .1, centered option activations/one-hot targets, 840 practice cases, no gradients. Final pass 25 repeats paired validation with unchanged C to verify restored artifact. Earlier validation C .54333 -> .73000, gain .18667 SE .02685. This repeat is not independent evidence.
