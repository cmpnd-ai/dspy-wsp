I’ll compare predicted and gold option frequencies within each question. I’ll also fix a stratified 450/150/300 practice split now: 450 records for deriving an edit, 150 for choosing its strength, and 300 reserved for an independent paired test.
```python
# Hypothesis: systematic option preferences, especially excessive danger/urgency, dominate over random errors.
stats = {}
baseline_by_id = {str(r["id"]): r for r in baseline}
def completion_probs(r):
    ans = {}
    for c in r["calls"]:
        vals = json.loads(c["completion"])
        for q,v in vals.items():
            ans[q] = {str(k):float(p) for k,p in v["probabilities"].items()} if "probabilities" in v else {"false":1-float(v["noul"]), "true":float(v["noul"])}
    return ans
for r in baseline:
    ex = by_id[str(r["id"])]
    wf = ex["inputs"]["workflow"]
    pp = completion_probs(r)
    for q,g in ex["labels"]["gold"].items():
        s = stats.setdefault((wf,q), {"n":0,"ok":0,"pred":collections.Counter(),"gold":collections.Counter(),"wrong_conf":[],"confusion":collections.Counter()})
        pred = max(pp[q], key=pp[q].get); target = max(g["probabilities"], key=g["probabilities"].get)
        s["n"]+=1; s["ok"]+=pred==target; s["pred"][pred]+=1; s["gold"][target]+=1
        s["confusion"][(target,pred)]+=1
        if pred!=target: s["wrong_conf"].append(pp[q][pred])
for (wf,q),s in stats.items():
    print(f"{wf[:8]:8} {q:22} acc={s['ok']/s['n']:.3f} wrongConf={np.mean(s['wrong_conf']):.2f}",
          "pred",dict(s["pred"]),"gold",dict(s["gold"]))
rng = np.random.default_rng(20260202)
splits = {"fit":[], "dev":[], "test":[]}
for i,wf in enumerate(sorted({ex["inputs"]["workflow"] for ex in examples})):
    ids = np.array([str(ex["id"]) for ex in examples if ex["inputs"]["workflow"]==wf], dtype=object)
    rng.shuffle(ids)
    nf = 113 if i<2 else 112
    nd = 37 if i<2 else 38
    splits["fit"] += ids[:nf].tolist()
    splits["dev"] += ids[nf:nf+nd].tolist()
    splits["test"] += ids[nf+nd:].tolist()
(WORK/"analysis_splits.json").write_text(json.dumps(splits))
print("Split sizes:", {k:len(v) for k,v in splits.items()})
print("Representative failures with evidence:")
for wf in sorted({ex["inputs"]["workflow"] for ex in examples}):
    cases = sorted((r for r in baseline if r["inputs"]["workflow"]==wf), key=lambda r:r["score"])
    for r in cases[:2]:
        print(r["id"],json.dumps(r["inputs"]["state"], ensure_ascii=False)[:1500],"\n",r["feedback"])
# Read the first 200 fitting records; this includes two workflows and 1000 questions.
read2_ids = splits["fit"][:200]
read2 = {"type":"read","model":"original","from":{"pass":1,"ids":read2_ids}}
(WORK/"read2_request.json").write_text(json.dumps(read2))
note("Question accuracy/frequencies:\n" + "\n".join(f"{wf}/{q}: {s['ok']}/{s['n']}; pred={dict(s['pred'])}; gold={dict(s['gold'])}" for (wf,q),s in stats.items()))
note("Fixed split seed 20260202: fit450/dev150/test300, stratified by workflow; IDs saved in analysis_splits.json. No fit or hyperparameter decisions will use test labels.\nPlanned forward 2: original READ first 200 fit IDs (exact list read2_request.json); 1000 questions. Edits: none. Inspect EOS states, normalized features, scalar and set correction.")
```
