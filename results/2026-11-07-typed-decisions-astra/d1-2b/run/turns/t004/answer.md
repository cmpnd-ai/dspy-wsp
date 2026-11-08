I’ll tabulate the errors and reserve a reproducible practice split before fitting anything. In particular, I want to check whether the model mistakes intended actions for observed outcomes, or systematically chooses the wrong severity level.
```python
# Hypothesis: observed-result and severity errors reflect misread state evidence rather than merely excessive confidence.
def call_probs(record):
    completion = json.loads(record["calls"][0]["completion"])
    out = {}
    for q, ans in completion.items():
        if "probabilities" in ans:
            out[q] = {str(k):float(v) for k,v in ans["probabilities"].items()}
        elif "noul" in ans:
            out[q] = {"false":1-float(ans["noul"]), "true":float(ans["noul"])}
    return out
base_by_id = {r["id"]:r for r in baseline}
stats, confusion, bad_records = {}, collections.defaultdict(collections.Counter), collections.defaultdict(list)
for r in baseline:
    wf = r["inputs"]["workflow"]
    for q,p in call_probs(r).items():
        g = by_id[r["id"]]["labels"]["gold"][q]["probabilities"]
        pred, target = max(p,key=p.get), max(g,key=g.get)
        key = (wf,q)
        s = stats.setdefault(key, [0,0,0.0,0.0])
        s[0] += 1
        s[1] += pred == target
        s[2] += p[pred]
        s[3] += sum((p[k]-g[k])**2 for k in g)
        if pred != target:
            confusion[key][(target,pred)] += 1
            bad_records[key].append((p[pred],r["id"]))
for key,s in sorted(stats.items(), key=lambda kv:kv[1][1]/kv[1][0]):
    print(f"{key[0]:27s} {key[1]:23s} acc={s[1]/s[0]:.3f} conf={s[2]/s[0]:.3f} brier={s[3]/s[0]:.3f} errors gold->pred {confusion[key].most_common(4)}")
rng = np.random.default_rng(71031)
split = {"fit":[], "dev":[], "test":[]}
for wf in sorted({e["inputs"]["workflow"] for e in examples}):
    ids = [e["id"] for e in examples if e["inputs"]["workflow"]==wf]
    rng.shuffle(ids)
    split["fit"] += ids[:90]
    split["dev"] += ids[90:135]
    split["test"] += ids[135:]
(WORK/"practice_split.json").write_text(json.dumps(split,indent=2))
print("Split sizes:", {k:len(v) for k,v in split.items()})
for key in [("agent_trace_observability","risk"),("agent_trace_observability","outcome"),
            ("customer_service","action"),("invoice_processing","urgency"),("security_incidents","severity")]:
    print("\nFAILURE EXAMPLES",key)
    for conf,eid in sorted(bad_records[key], reverse=True)[:2]:
        r = base_by_id[eid]
        print(eid, "state:",json.dumps(r["inputs"]["state"])[:1800])
        print("pred",call_probs(r)[key[1]],"gold",by_id[eid]["labels"]["gold"][key[1]]["probabilities"])
summary = "\n".join(f"- {wf}/{q}: acc {s[1]/s[0]:.3f}; gold->pred {confusion[(wf,q)].most_common(3)}" for (wf,q),s in sorted(stats.items()))
note("Baseline breakdown:\n"+summary+"\n\nPractice split fixed with RNG 71031, stratified: fit=360, dev=180, test=360. Split saved to practice_split.json. Diagnostic tables use all baseline records, but weight fitting will use fit only; dev for strength selection; test for independent paired evaluation.")
```
