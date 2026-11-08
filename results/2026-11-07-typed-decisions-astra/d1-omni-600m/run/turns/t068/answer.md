The prior-only ablation scored 70.35% on the reserved set, below the full repair’s 72.46%. I’ll restore the full repair and use the remaining diagnostic passes to check whether adding an irrelevant record identifier disrupts its decisions. No further fitting or tuning will be done.
```python
# Hypothesis: the frozen repair reads substantive evidence and should remain stable when irrelevant record metadata is added.
prior_test={r["id"]:r for r in map(json.loads,open(W/"gen_22_edited.jsonl"))}
state_test_diff=np.array([te[i]["score"]-prior_test[i]["score"] for i in splits["test"]])
print("Full minus prior-only, reserved test:",
      {"gain":float(state_test_diff.mean()),"se":float(state_test_diff.std(ddof=1)/len(state_test_diff)**.5),
       "improved":int((state_test_diff>0).sum()),"worsened":int((state_test_diff<0).sum())})
save_scorer_params(best2_params)
robust_ids=[]
for wf in workflows:
    robust_ids.extend([i for i in splits["test"] if by_id[i]["inputs"]["workflow"]==wf][:25])
robust_requests,robust_tags=[],[]
for j,iid in enumerate(robust_ids):
    req=json.loads(to[iid]["calls"][0]["prompt"])
    assert isinstance(req["state"],dict)
    for variant in ["unchanged","irrelevant_reference"]:
        rr=copy.deepcopy(req)
        if variant=="irrelevant_reference":
            rr["state"]["record_reference"]=f"RECORD-{700001+137*j}"
        robust_requests.append(rr)
        robust_tags.append({"id":iid,"variant":variant,"workflow":by_id[iid]["inputs"]["workflow"]})
json.dump({"type":"read","model":"original","requests":robust_requests},open(W/"metadata_robustness_original.json","w"))
json.dump({"type":"read","model":"edited","requests":robust_requests},open(W/"metadata_robustness_edited.json","w"))
json.dump(robust_tags,open(W/"metadata_robustness_tags.json","w"))
note("\nPass22 prior-only reserved test:58.421% ->70.351%, +11.930pp SE1.618pp. Full-minus-prior-only="+str(float(state_test_diff.mean()))+" SE="+str(float(state_test_diff.std(ddof=1)/len(state_test_diff)**.5))+". No refitting.\nRestored frozen best2 exactly (conditional_candidate.pt). Next pass23: READ original200 requests/1000questions in metadata_robustness_original.json: first25 reserved examples per workflow, each unchanged and with irrelevant record_reference added. Selection by pre-existing split order, not outcomes. Installed edits are best2, but original requested. Preregistered expectation: substantive answers and the repair's gain should be largely stable. Pass24 will replay edited; pass25 final validation.")
print("Prepared robustness requests",len(robust_requests),"from",len(robust_ids),"reserved examples.")
```
