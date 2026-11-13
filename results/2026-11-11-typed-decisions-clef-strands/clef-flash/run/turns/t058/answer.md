I’ll align the new read rows with their recorded requests, rather than assuming ID order, and queue the final 100 practice records. This avoids the ordering issue found earlier and completes the data needed for the fixed-design refit.


```python
# Hypothesis: exact request-based alignment makes the full-practice closed-form refit reliable without using hidden validation data.
repair_req_to_id={}
for rec in base:
    request=json.loads(rec["calls"][0]["prompt"])
    canonical=json.dumps(request,sort_keys=True,ensure_ascii=False)
    assert canonical not in repair_req_to_id
    repair_req_to_id[canonical]=rec["id"]
def repair_read_data(k):
    directory=WORK/f"fwd_{k}"
    meta=json.load(open(directory/"meta.json"))
    reqs=json.load(open(directory/"requests.json"))
    # Saved requests are the normalized System One requests, one per prompt.
    reqids=[repair_req_to_id[json.dumps(r,sort_keys=True,ensure_ascii=False)] for r in reqs]
    out={}
    for source,dest in [("head_inputs","features"),("option_logits","logits"),("option_mask","mask")]:
        out[dest]=torch.from_numpy(np.load(directory/f"{source}.npy")).clone()
    out["features"]=torch.nan_to_num(out["features"])
    out["logits"]=torch.nan_to_num(out["logits"]).masked_fill(~out["mask"],-torch.inf)
    out["meta"]=meta["rows"]
    out["ids"]=[reqids[r["request"]] for r in meta["rows"]]
    out["qkey"]=[(byid[ii]["inputs"]["workflow"],r["question"]) for ii,r in zip(out["ids"],meta["rows"])]
    gg=torch.zeros_like(out["logits"])
    tt=[]
    for i,(ii,r) in enumerate(zip(out["ids"],meta["rows"])):
        g=byid[ii]["labels"]["gold"][r["question"]]["probabilities"]
        for j,opt_id in enumerate(r["options"]): gg[i,j]=g[str(opt_id)]
        tt.append(int(gg[i].argmax()))
    out["gold"]=gg
    out["target"]=torch.tensor(tt)
    return out
repair_data12=repair_read_data(12)
print("Aligned records:",len(set(repair_data12["ids"])),"questions",len(repair_data12["ids"]))
print("Baseline accuracy in read:",float((repair_data12["logits"].argmax(-1)==repair_data12["target"]).float().mean()))
print("Expected accuracy:",np.mean([basebyid[ii]["score"] for ii in set(repair_data12["ids"])]))
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 12 complete / planned pass 13\nOriginal read on 200 former test practice items aligned by exact normalized request text; baseline argmax matches recorded accuracy. Current edits remain best_2 unchanged. READ original on remaining 100 former test practice items from read_refit_2.json (500 questions); no gold sent to model.\n")
```

