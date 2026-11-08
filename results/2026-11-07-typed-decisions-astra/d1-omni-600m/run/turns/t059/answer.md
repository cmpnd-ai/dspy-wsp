The frozen repair improved the reserved practice set from 58.42% to 72.46%: **+14.04 percentage points, paired SE 1.65 points**. I’ll examine its question-level gains and read a fixed subset’s scorer inputs to check gate selectivity and contribution sizes out of sample—not to refit the edit.
```python
# Hypothesis: the frozen repair's independent gain comes from improved evidence readout across workflows, with selective gates and negligible displaced-neuron effects.
test_orig=[json.loads(s) for s in open(W/"gen_19_original.jsonl")]
test_edit=[json.loads(s) for s in open(W/"gen_19_edited.jsonl")]
to={r["id"]:r for r in test_orig}
te={r["id"]:r for r in test_edit}
print("Record keys",list(test_edit[0]))
print("Parsed probability example",probs_record(test_edit[0]))
for wf in workflows:
    wi=[i for i in splits["test"] if by_id[i]["inputs"]["workflow"]==wf]
    diff=np.array([te[i]["score"]-to[i]["score"] for i in wi])
    print(wf,len(wi),"original",round(np.mean([to[i]["score"] for i in wi]),4),
          "edited",round(np.mean([te[i]["score"] for i in wi]),4),
          "gain",round(diff.mean(),4),"SE",round(diff.std(ddof=1)/len(diff)**.5,4))
test_read_ids=splits["test"][:200]
json.dump({"type":"read","model":"original","from":{"pass":19,"ids":test_read_ids,"model":"original"}},open(W/"frozen_test_read.json","w"))
note("\nPASS19 INDEPENDENT TEST: frozen best2 original58.421% ->72.456%, gain14.035pp paired SE1.653pp,131 improved/37 worsened among228 reserved practice examples. No test fitting or selection will be done.\nNext pass20: READ original first200 reserved test IDs in frozen_test_read.json (1000questions), frompass19original. Installed edit unchanged best2/conditional_candidate.pt. Purpose only gate-selectivity and displaced-neuron checks and frozen ablations; no weight fitting.")
```
I’ll inspect workflow-level gains and prepare a read-only diagnostic on 200 reserved examples. The edit is frozen; these results will not be used to fit or tune weights.
```python
# Hypothesis: the frozen repair generalizes across workflows, rather than exploiting a single question's answer prior.
test_orig=[json.loads(s) for s in open(W/"gen_19_original.jsonl")]
test_edit=[json.loads(s) for s in open(W/"gen_19_edited.jsonl")]
to={r["id"]:r for r in test_orig}
te={r["id"]:r for r in test_edit}
print("Record keys",list(test_edit[0]))
for wf in workflows:
    wi=[i for i in splits["test"] if by_id[i]["inputs"]["workflow"]==wf]
    diff=np.array([te[i]["score"]-to[i]["score"] for i in wi])
    print(wf,len(wi),"original",round(np.mean([to[i]["score"] for i in wi]),4),
          "edited",round(np.mean([te[i]["score"] for i in wi]),4),
          "gain",round(diff.mean(),4),"SE",round(diff.std(ddof=1)/len(diff)**.5,4))
test_read_ids=splits["test"][:200]
json.dump({"type":"read","model":"original","from":{"pass":19,"ids":test_read_ids,"model":"original"}},open(W/"frozen_test_read.json","w"))
note("\nPASS19 INDEPENDENT TEST: frozen best2 original58.421% ->72.456%, gain14.035pp paired SE1.653pp,131 improved/37 worsened among228 reserved practice examples. No test fitting or selection will be done.\nNext pass20: READ original first200 reserved test IDs in frozen_test_read.json (1000questions), frompass19original. Installed edit unchanged best2/conditional_candidate.pt. Purpose only gate-selectivity and displaced-neuron checks and frozen ablations; no weight fitting.")
```
