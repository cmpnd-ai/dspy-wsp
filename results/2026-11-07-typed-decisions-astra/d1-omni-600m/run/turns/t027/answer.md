Validation confirms the repair: accuracy rises from 57.53% to 68.80%, a gain of 11.27 points with paired SE 1.52. I’ll preserve this version and verify its counterfactual effects in an actual edited forward pass before investigating any remaining weaknesses.
The next block backs up the confirmed weights, summarizes the remaining development errors, and prepares the counterfactual replay with the same edit.
```python
# Hypothesis: the validated residual channel repairs specific state-cue responses, while remaining errors may reveal cross-question interference.
best1=WORK/"best_1"
if best1.exists():shutil.rmtree(best1)
shutil.copytree(WORK/"edited",best1)
best1_manifest={"name":"linear residual channel","validation_pass":9,"gain":0.11266666666666668,
                "se":0.01519544417242141,"original":0.5753333333333334,"edited":0.688,
                "definition":"candidate_full_definition.pt"}
json.dump(best1_manifest,open(WORK/"best_1_manifest.json","w"),indent=2)
ed8=[json.loads(s) for s in (WORK/"gen_8_edited.jsonl").read_text().splitlines()]
ed8_by_id={str(r["id"]):r for r in ed8}
current_conf=collections.defaultdict(collections.Counter)
for i,r in ed8_by_id.items():
    pred=probs_record(r)
    for q,g in by_id[i]["labels"]["gold"].items():
        p=max(pred[q],key=pred[q].get);t=max(g["probabilities"],key=g["probabilities"].get)
        current_conf[(r["inputs"]["workflow"],q)][t,p]+=1
for (wf,q),c in current_conf.items():
    acc=sum(v for (t,p),v in c.items() if t==p)/sum(c.values())
    print(wf[:8],q,round(acc,3),"errors",collections.Counter({k:v for k,v in c.items() if k[0]!=k[1]}).most_common(2))
json.dump({"type":"read","model":"edited","requests":cf_requests},open(WORK/"counterfactual_edited.json","w"))
note("PASS 9 VALIDATION CONFIRMED best_1: 300 hidden examples, original 57.533%, edited 68.800%, +11.267pp, paired SE 1.520pp; 157 improved / 59 worsened. Backup edited directory to best_1/ and manifest. This is the current best confirmed edit.\n"
     "Next pass: READ edited on the identical 144 controlled counterfactual requests in counterfactual_edited.json (720 questions). Exact edit: full-strength candidate_linear_full, same as validation. Preregistered increases: violation->harmful contrast .0945 -> .4163, risk .4106 -> 1.0435, overdue urgency .0429 -> .3792. Discount and freight not repaired by this channel.")
```
