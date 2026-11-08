The best confirmed edit is now strength 0.875: validation accuracy rose from 34.80% to 58.07%, a paired gain of 23.27 ± 1.52 percentage points. Its advantage over strength 0.75 is too small to claim as significant. I’ll freeze it and verify program outputs across all practice records, explicitly treating those as fitted data.
```python
# Hypothesis: the frozen scalar repair should match cached head predictions exactly when run through the full DSPy program.
note("\nForward19 strength.875 hidden validation300: original .348000, edited .580667, paired gain .232667 SE .015184;207 improved/30 worsened. Best confirmed aggregate score. Difference +.001333 vs strength.75 is not claimed significant. Candidate selection now frozen.")
if not (work/"best_4").exists():
    shutil.copytree(work/"edited",work/"best_4")
frozen_weight = load("scalar.weight").clone()
torch.save({"scalar.weight":frozen_weight},work/"best_4_weights.pt")
final_direction = (frozen_weight-load_original("scalar.weight")).numpy().reshape(-1)
orig_w = load_original("scalar.weight").numpy().reshape(-1)
scale_component = float(final_direction@orig_w/(orig_w@orig_w))
non_scale_fraction = float(np.linalg.norm(final_direction-scale_component*orig_w)/np.linalg.norm(final_direction))
final_cv_logits = L900+.875*cross_correction
print("Final recipe cross-fit:",evaluate900(final_cv_logits))
print("Final cross-fit versus prior-only:",paired900_between(final_cv_logits,L900+.875*cv_prior_corr))
print("Correction norm:",float(np.linalg.norm(final_direction)),
      "non-scaling norm fraction:",non_scale_fraction,
      "cosine with original readout:",float(final_direction@orig_w/(np.linalg.norm(final_direction)*np.linalg.norm(orig_w))))
all900_request = {"type":"decide","model":"edited","problems":[ex["id"] for ex in examples]}
(work/"final_practice900_request.json").write_text(json.dumps(all900_request))
for wf in sorted(set(wf900)):
    example = next(ex for ex in examples if ex["inputs"]["workflow"]==wf)
    print("State sample",wf,json.dumps(example["inputs"]["state"],ensure_ascii=False)[:1800])
note("\nPlanned forward20: edited DECIDE all900 practice IDs in original trainset order, final_practice900_request.json. Exact edit frozen best4 scalar.weight=w0+.875*d_mix900 (candidate_weights/mix900_0875_scalar.pt); all other tensors original. Compare deterministic outputs with forward1 and cached full900 head calculation. This is in-sample verification, not independent evidence.")
```
 tool call duplication? We have only one block good.The frozen edit is preserved as `best_4`; the next check is full-program agreement with the cached head calculation.
