# Weight repair session
No edits. Stratified random split seed 718; fit=448, holdout=452.
Next pass: original DECIDE on all 900 practice examples.
Baseline 900 practice accuracy .61333 SE .00825. Inspecting question confusions. Next READ original: 200 fit cases (1000 questions), last layer inputs; no edits.
Pass 3 READ: causal probe 1, last post_ffn_norm = g0 + random-sign*g0 (seed 851), same 200 fit requests as pass 2. No other edits.
Pass 4 READ: causal probe 2, last post_ffn_norm = g0 + second independent random-sign*g0, same 200 fit cases. No other edits.
Two causal probes reconstructed branch B; gain solutions use 384 largest causally sensitive dimensions, closed-form ridge vs gold log probabilities. Probes removed.
Alignment fixed by exact state matching: READ from ids follows baseline order. Next pass 5: half-strength ridge=1 correction on 384 last-post-FFN gain dimensions; DECIDE both on 448 fit cases only. Direction stored gain_direction.npy.
Pass 5 half-strength: fit gain +.039286 SE .005523 (103 improved,28 worsened). Next pass 6: strength 1.0, same ridge=1 384-dim direction, same 448 fit cases; dose-response.
Pass 6 strength1 fit +.052232 SE .007295. Next pass7 strength2 same direction and same 448 fit cases. Inspect question-specific gains to verify non-global effect.
Pass7 strength2 fit gain +.067411 SE .009955. Next pass8 original READ on 200 additional fit-only cases, no edits. Holdout untouched.
Pass9 READ additional 200 fit cases with causal probe1 g0+d1; no other edits.
Pass10 READ additional 200 fit cases with probe2 g0+d2; no other edits.
Pass11: candidate B, ridge1 closed-form solve on 400 fit cases, 384 causally sensitive dimensions, strength1; DECIDE both all448 fit. No holdout used.
Pass11 candidate B strength1 fit +.054464 SE .007568. Pass12 candidate B strength2 on same448 fit.
Pass12 B strength2 fit +.069643 SE .010344. Next pass13 candidate C ridge.1, same400 fit causal measurements, same384dims, strength1; DECIDE both448 fit.
Pass13 C strength1 fit +.072321 SE .008402. Pass14 C strength2 same448 fit.
Pass14 C strength2 fit +.080357 SE .011409. Pass15 C strength.5 same448 fit, lower-dose bracket.
Pass15 C strength.5 fit +.05625 SE .006039. Pass16 C strength3 same448 fit, upper-dose bracket; predicted below strength2.
Pass16 C strength3 fit +.00625 SE .014045, saturation confirmed. LOCK selected candidate C strength2 (ridge.1,384dims,400 fit cases). Pass17 independent practice holdout450 of452; no further fitting or strength selection on holdout.

Pass17 independent practice holdout: original .619111, edited .684444, gain +.065333 SE .011405; 192 improved,107 worsened (450 items). Candidate remains LOCKED; no holdout-based selection.
Next pass18 VALIDATION both, all300 hidden examples. Exact edit: layers.23.post_ffn_norm.weight = original + 2*candidate-C direction (ridge .1,384 dimensions,400 fit examples); all other tensors unchanged. Backup locked_C_strength2.
Pass18 VALIDATION confirms locked edit: original .606667, edited .679333, gain +.072667 SE .013075;123 improved,58 worsened/300. KEEP candidate C strength2. Remaining passes diagnostic only; no further selection.
Pass19 diagnostic READ original, first150 pass17 holdout records (750 questions). Validated C-strength2 stays installed; original READ bypasses edits. No fitting.
Pass20 diagnostic READ edited, same first150 holdout records as pass19. Exact edit remains locked C-strength2 on384 last-post-FFN gains; no fitting.
Pass21 READ original next150 pass17 holdout records; current edit unchanged locked C-strength2. Diagnostic only.
Pass22 READ edited next150 holdout, matched pass21; locked C-strength2 unchanged. Diagnostic only.
Pass23 READ original final150 holdout; locked C-strength2 remains installed, no changes or fitting.
Pass24 READ edited final150 holdout matched pass23; exact locked C-strength2 unchanged. Diagnostic only.
Full450 holdout readout audit: 700/2250 top-option changes; confidence increases 503, decreases 1747. Mean rounded-feedback Brier 0.16158 -> 0.11017; paired delta SE 0.00497. Only384 entries in last post_ffn_norm changed, finite.
Pass25 final DECIDE edited all900 practice examples for reporting only; locked validated candidate C-strength2 unchanged. No further selection.
