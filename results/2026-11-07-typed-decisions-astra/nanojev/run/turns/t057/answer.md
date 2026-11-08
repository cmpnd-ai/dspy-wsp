The full program scored 66.98% on the fitted practice set. I’ll check exact agreement with cached predictions, then temporarily keep only the average question/option correction. Comparing that ablation with the frozen edit tests how much the state-varying component contributes.
```python
# Hypothesis: removing the state-varying part of the frozen correction should erase a substantial share of its gains.
final_records = [json.loads(line) for line in (work/"gen_20.jsonl").read_text().splitlines()]
final_by_id = {r["id"]:r for r in final_records}
print("Record keys:",list(final_records[0]))
print("Completion-probability sample:",completion_probs(final_records[0]))
frozen_logits900 = L900+np.einsum("qod,d->qo",X900,final_direction)
print("Cached final fitted score:",evaluate900(frozen_logits900))
print("Program score sample:",{k:v for k,v in final_records[0].items()
                               if k in ["score","feedback","metric"]})
program_diffs = np.array([float(final_by_id[i]["score"])-float(baseline_by_id[i]["score"]) for i in unique900])
print("Actual all900 fitted comparison:",
      {"gain":float(program_diffs.mean()),"se":float(program_diffs.std(ddof=1)/np.sqrt(900)),
       "improved":int((program_diffs>0).sum()),"worsened":int((program_diffs<0).sum())})
note("\nForward20 final frozen best4 all900 fitted practice: original .364222, edited .669778; paired gain "+
     str(float(program_diffs.mean()))+" SE "+str(float(program_diffs.std(ddof=1)/np.sqrt(900)))+
     ". This is in-sample, not a held-out gain. Cached scalar-head prediction matches aggregate program accuracy.")
ablation_ids = splits["dev"]+splits["test"]
ablation_selection = np.isin(ids900,ablation_ids)
prior900_logits = L900+.875*np.einsum("qod,d->qo",X900,dp900)
print("Prior-only expected on450:",evaluate900(prior900_logits,ablation_selection))
print("Full expected on450:",evaluate900(frozen_logits900,ablation_selection))
wprior900 = torch.from_numpy(w0+.875*dp900).reshape(1,-1)
torch.save(wprior900,work/"candidate_weights"/"mix900_prior_only_scalar.pt")
save_edit("scalar.weight",wprior900)
ablation_request = {"type":"decide","model":"both","problems":ablation_ids}
(work/"final_prior_ablation_request.json").write_text(json.dumps(ablation_request))
note("\nPlanned forward21: paired DECIDE dev150+retired-test300 (450 IDs in final_prior_ablation_request.json). Exact temporary edit scalar.weight=w0+.875*dp900, candidate_weights/mix900_prior_only_scalar.pt; all other tensors original. Mechanism-only ablation on fitted practice records, not independent accuracy evidence. Restore frozen best4 immediately afterward.")
```
