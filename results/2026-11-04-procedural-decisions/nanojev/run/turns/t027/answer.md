The prior-only intervention gives essentially no gain: +0.22 points, SE 2.94. I’ll restore the validated full edit and test matched counterfactual records—changing one decisive fact at a time—to see whether the repaired head responds to the rule-relevant cues.

```python
# Hypothesis: the repaired scalar direction responds to decisive facts, not just answer names or workflow identity.
import copy
save_edit("scalar.weight",torch.from_numpy((W0+delta900).astype(np.float32)[None]))
probe_requests=[]; probe_pairs=[]
def addpair(r,q,s0,s1,opts):
    start=len(probe_requests)
    for s in [s0,s1]:
        prompt=json.loads(r["calls"][0]["prompt"])
        prompt["state"]["inputs"]["state"]=s
        prompt["questions"]={q:prompt["questions"][q]}
        probe_requests.append(prompt)
    probe_pairs.append({"question":q,"rows":[start,start+1],"targets":opts,"source":r["id"]})
rng=random.Random(834)
for q in stats:
    candidates=[r for r in records if q in byid[r["id"]]["labels"]["questions"]]
    rng.shuffle(candidates)
    for r in candidates[:30]:
        a=copy.deepcopy(r["inputs"]["state"]); b=copy.deepcopy(a)
        if q=="current_owner":
            ts=max([e["ts"] for e in a["events"]]+[0])+10
            a["events"].append({"ts":ts,"kind":"assign","owner":"alice"})
            b["events"].append({"ts":ts,"kind":"assign","owner":"bob"})
            targets=["alice","bob"]
        elif q=="current_severity":
            ts=max([e["ts"] for e in a["events"]]+[0])+10
            a["events"].append({"ts":ts,"kind":"severity","severity":"routine"})
            b["events"].append({"ts":ts,"kind":"severity","severity":"critical"})
            targets=["0","2"]
        elif q=="is_open":
            ts=max([e["ts"] for e in a["events"]]+[0])+10
            a["events"].append({"ts":ts,"kind":"resolve"})
            b["events"].append({"ts":ts,"kind":"reopen"})
            targets=["false","true"]
        elif q=="intent":
            for s in [a,b]:
                s["billing"]={k:False for k in s["billing"]}
                s["account"].update(active=False,auth_failures=0,permission_mismatch=False)
                s["telemetry"].update(integration_failures=3,error_rate_percent=30)
            b["billing"]["duplicate_charge"]=True
            targets=["technical","billing"]
        elif q=="is_urgent":
            a["timeline"]={"deadline_hours":72,"executive_escalation":False}
            b["timeline"]={"deadline_hours":12,"executive_escalation":False}
            targets=["false","true"]
        elif q=="workflow_impact":
            for s in [a,b]:
                s["workflow"]={"affected_features":[{"feature":"checkout","status":"restored"}],
                               "core_features":["checkout"],"workarounds":[]}
            b["workflow"]["affected_features"][0]["status"]="down"
            targets=["0","2"]
        else:
            for s in [a,b]:
                s["prior_probability_incident"]="1/2"
                s["sensors"]=[{"id":"S1","observed":"negative","p_positive_given_incident":"4/5",
                               "p_positive_given_no_incident":"1/5"}]
            b["sensors"][0]["observed"]="positive"
            targets=["false","true"]
        addpair(r,q,a,b,targets)
print("Probe pairs",len(probe_pairs),"questions",len(probe_requests))
print("Owner event kinds",collections.Counter(e["kind"] for r in records[:300] for e in r["inputs"]["state"]["events"]))
json.dump({"type":"read","model":"original","requests":probe_requests},open(work/"probe_original.json","w"))
json.dump({"type":"read","model":"edited","requests":probe_requests},open(work/"probe_edited.json","w"))
with open(notes,"a") as f:
    f.write("\nPass 10 prior-only control: .47185 -> .47407, gain .00222 ± .02939. Restored validated delta900."
            "\nPasses 11-12: 210 matched cue pairs (30 per question), original then edited read. One decisive fact changed per pair; event owner probe uses final assign event. No probe used to fit.\n")
```
