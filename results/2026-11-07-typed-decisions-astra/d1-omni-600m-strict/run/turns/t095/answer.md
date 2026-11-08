The gate preserved agent accuracy, but it did not improve the overall metric: **−0.20 points ± 0.45 paired SE** versus the validated checkpoint. I’ll reject it. With three passes left, I’ll audit the retained repair’s cue response and performance rather than add an unsupported change.
```python
# Hypothesis: the gate localizes the violation effect but offers no accuracy gain, so best_2 should remain the final candidate.
for target22 in [True,False]:
    changes22=[]
    remove_residual22=[]
    changes_rank22=0
    for r22,r18 in zip(rowinfo22,rowinfo18):
        if (r22["wf"]=="agent_trace_observability") != target22:
            continue
        i22=r22["i"]; n22=len(r22["opts"])
        change22=np.asarray(arr22["option_logits"][i22,:n22]-arr18["option_logits"][i22,:n22])
        old_effect22=np.asarray(arr10["option_logits"][i22,:n22]-arr8["option_logits"][i22,:n22])
        change22=change22-change22.mean()
        old_effect22=old_effect22-old_effect22.mean()
        changes22.extend(change22)
        remove_residual22.extend(change22+old_effect22)
        changes_rank22+=int(r22["pi"]!=r18["pi"])
    print("AGENT" if target22 else "OTHER",
          "centered delta RMS",float(np.sqrt(np.mean(np.square(changes22)))),
          "residual after removing old violation effect",float(np.sqrt(np.mean(np.square(remove_residual22)))),
          "changed rankings",changes_rank22)
backup_edits("agent_gated_candidate")
print("BEST_2 FILES",[(p.name,p.stat().st_size) for p in (work/"best_2").iterdir()])
print("REQUEST FILE SCHEMAS")
for name in ["deadline_counterfactuals.json","agent_counterfactuals.json","security_counterfactuals.json"]:
    obj22=json.loads((work/name).read_text())
    print(name, list(obj22),len(obj22.get("requests",[])))
with open(work/"notes.md","a") as f:
    f.write("\nPass22 agent-workflow gate: discovery B .593 vs original .557 "
            "(+.0360 SE .010101); versus best_2 -.0020 SE .004481. "
            "Agent predictions unchanged, but removing off-workflow effects does not improve accuracy. "
            "Reject candidate; retain validated best_2. Remaining three passes reserved for "
            "matched-cue audit, untouched reserve cases and final discovery evaluation.\n")
```
