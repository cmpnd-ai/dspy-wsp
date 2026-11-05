# Weight-repair experiment
No edits yet. Plan: baseline all 900 practice cases; inspect errors and read decision-head representations. Reserve a deterministic half of practice for testing closed-form edits.

Pass 1 planned: original weights, all 900 practice examples; no edits.

Baseline: 0.56556 ± 0.00714 (900 cases). Fixed random seed 42 split: 450 fit, 450 independent practice test. Read passes planned in five batches of 180 (900 questions each); original, no edits.

Scorer reconstruction max error 1.91e-6. Closed-form centered GELU-feature residual regression, 450 fit cases only, target calibrated log gold (floor .015). Ridge 10: fit +.0604±.0087, reserved practice +.0636±.0094; ridge100 +.0284±.0067 test, ridge1000 +.0067±.0038. No saved edit yet. Exploring lower regularization using fit cases only before causal dose tests.

Fit-only 3-fold regularization selection: {0.01: np.float64(0.6293), 0.1: np.float64(0.6333333333333333), 1: np.float64(0.6355333333333333), 3: np.float64(0.6248666666666666), 10: np.float64(0.6138)}; selected ridge 1. Current edit: head.scorer.3.weight = original + 0.5 * saved readout_delta.npy, no other edits. Next pass both on 450 reserved practice cases; dose-response plan .5, 1, 1.5.

Pass7 half-dose causal test: reserved 450 original .56133, edited .60756, gain .04622±.00720; 135 improved/59 worsened. Exact local prediction confirmed. Current edit full-dose ridge1 readout correction. Next pass both on same 450.

Pass8 full-dose: reserved gain .08000±.01094, .56133→.64133; 208 improved/100 worsened. Current edit 1.5-dose; next same reserved 450, predicted overshoot gain .0258.

Pass9 1.5-dose: gain .02578±.01293, confirming overshoot. Restored full-dose ridge1 scorer readout. Next validation both, all 300 unseen cases; no validation labels used in fitting.

Pass10 VALIDATION confirms full-dose: .57533→.64533, gain .07000±.01302, 133 improved/63 worsened. Saved best_1 and candidate_full. Current best = ridge1 full-dose final scorer weight only.

Cue counterfactual reads planned original+edited: 9 trace requests (3 bases × violations 0/1/2) and 6 invoice requests (2 bases × due 30/2/-21). Current edit full-dose ridge1 final readout only. Purpose: test cue sensitivity independent of right/wrong-case difficulty.

Pass11/12 matched cue reads completed; inputs should be identical because readout only changes. Current trial: same closed-form residual readout refit on all 900 practice cases, ridge2 (same per-observation strength); delta saved readout_delta_all.npy. Next both on first 450 practice cases, then remaining 450, then validation. Best_1 retains independently confirmed 450-fit edit.

All900 refit passes13/14 in-sample practice gain 0.08933±0.00740. Current edit all900 ridge2 readout. Next validation both on 300.

Pass15 all900 refit VALIDATION .57533→.65600, +.08067±.01260; 142 improved/60 worsened. Saved best_all. Diagnostic ablation: project all900 delta onto span of workflow/question/option mean centered features (prior component); complementary context component. Next pass both on 450 reserved practice (now used in all900 fit; diagnostic only), prior-only edit.

Pass16 prior-only ablation gain .02044±.01104. Current diagnostic edit context-only complementary projection; next both same 450 cases. Best remains full all900 in best_all.

Pass17 context-only ablation gain .04133±.00827; supports contextual discrimination beyond prior correction. Current diagnostic half-strength all900 edit. Next both same450. Plan also 1.5 strength, sparse neuron ablations; do not replace best without validation.

Pass18 half-dose all900: +.05289±.00752 on diagnostic 450. Current edit 1.5*all900 delta; next same450 paired dose test. Best remains validated full all900.

Pass19 1.5-dose +.05022±.01304 vs full +.09289 on same450; overshoot confirmed. Restored full all900 delta for diagnostic reads. Next inspect recorded cue responses and workflow gains.

Diagnostics: all900 practice largest gains trace outcome +.209, risk +.200; customer churn +.191; invoice urgency +.178; security severity +.173. Cue reads have identical hidden states but mixed cue improvements: invoice overdue urgency sensitivity increases; trace review sensitivity not consistently improved. No universal cue-repair claim. Current prior-only projection of all900 correction; next validation causal ablation (not candidate selection).

Pass20 prior-only VALIDATION +.00067±.01262, effectively no benefit. Current context-only complement; next validation ablation. Full all900 remains selected best.

Pass21 context-only VALIDATION +.02667±.00987; full combination best +.08067±.01260. Current top512 magnitude entries of all900 correction only; next diagnostic same450 practice. Then complementary512 ablation, full-edit read, final validation.

Pass22 top512 by absolute coefficient correction: diagnostic450 +.00444±.01303 (not the earlier activation-weighted ranking). Current complementary512 correction; next same450. Evidence favors distributed balanced readout, not a few largest coefficients.

Pass23 complementary512 +.00178±.00209; both halves alone ineffective versus full +.09289 diagnostic450. Restored final full all900 ridge2 delta, head.scorer.3.weight ONLY. Pass24 planned edited READ on 15 matched cue requests via cues_edited.json; pass25 final validation confirmation. No further candidate selection.

Pass24 final edited cue READ: identical head inputs; logit changes reconstructed directly from scorer GELU features (kernel output). Final pass25 validation both on300. Exact current edit full all900 ridge2 correction ONLY to head.scorer.3.weight. Validation used for confirmation, never deriving coefficients.
