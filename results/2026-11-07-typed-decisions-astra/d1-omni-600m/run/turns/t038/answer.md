The conditional extension is confirmed: validation rises to 71.73%, a gain of 14.20 points over original with paired SE 1.54. I’ll preserve it and test whether its added benefit comes from state-sensitive corrections rather than fixed option preferences.
The next block compares state-dependent and fixed-prior ablations locally, then prepares an actual replay of the controlled cue changes.
```python
# Hypothesis: conditional channels help by interpreting state cues, not merely by adding fixed question/option priors.
best2=WORK/"best_2"
if best2.exists():shutil.rmtree(best2)
shutil.copytree(WORK/"edited",best2)
json.dump({"validation_pass":13,"gain":.142,"se":.015428043498721386,"score":.7173333333333334,
           "definition":"conditional_candidate.pt"},open(WORK/"best_2_manifest.json","w"),indent=2)
prior_add=torch.zeros_like(D["logits"])
state_add=torch.zeros_like(D["logits"])
for qno,o in selected_gate_keys:
    aq=qid==qno; fq=D["fit"]&aq
    d,b=conditional_solutions[qkeys[qno]][o][.1]
    avg=float((D["x"][fq,o]@d+b).mean())
    prior_add[aq,o]=avg
    state_add[aq,o]=D["x"][aq,o]@d+b-avg
print("Full conditional",evaluate_logits(base_linear+selected_add))
print("Fixed prior only",evaluate_logits(base_linear+prior_add))
print("State-dependent only",evaluate_logits(base_linear+state_add))
print("Targeted residual variability (RMS): prior",float(prior_add[D["dev"][:,None]&D["mask"]].square().mean().sqrt()),
      "state",float(state_add[D["dev"][:,None]&D["mask"]].square().mean().sqrt()))
json.dump({"type":"read","model":"edited","requests":cf_requests},open(WORK/"counterfactual_conditional.json","w"))
note("PASS13 VALIDATION CONFIRMED best_2: original57.533% ->71.733%, +14.200pp SE1.543pp,171 improved/47 worsened. Best2 backup created; candidate conditional_candidate.pt. Conditional extension improves validation aggregate +2.933pp over best1 (no paired SE between hidden candidates available).\n"
     "Next pass: READ edited identical144 counterfactual requests,720questions, in counterfactual_conditional.json. Exact current edit unchanged conditional_candidate.pt (110 neurons). Predicted effects: discount urgency contrast .0064->.3046, prohibited freight discrepancy .0507->.1887, overdue urgency .3792->.6032, violations harmful .4163->.5368. Compare with passes6 and10.")
```
