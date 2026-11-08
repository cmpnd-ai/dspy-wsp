Before fitting anything, I’ll verify row-to-example alignment against the recorded probabilities. Then I’ll finish the development read; a mislabeled activation row would otherwise produce a misleading repair.
```python
# Hypothesis: final-scorer option contrasts retain the information needed for a small, state-sensitive closed-form repair.
def load_read_dataset(passno, ids):
    folder=WORK/f"fwd_{passno}"
    meta=json.load(open(folder/"meta.json"))
    selected=set(ids)
    # The 'from' tool retains generation-file order, rather than the order in its ID filter.
    ordered=[str(r["id"]) for r in baseline if str(r["id"]) in selected]
    arrays={p.stem:np.load(p,mmap_mode="r") for p in folder.glob("*.npy")}
    rows=[]
    gold=np.zeros_like(arrays["option_logits"])
    expected=np.zeros_like(gold)
    for j,m in enumerate(meta["rows"]):
        i=ordered[m["request"]]; q=m["question"]
        n=len(m["options"])
        assert q in by_id[i]["labels"]["gold"]
        gold[j,:n]=[by_id[i]["labels"]["gold"][q]["probabilities"][o] for o in m["options"]]
        expected[j,:n]=[pbase[i][q][o] for o in m["options"]]
        rows.append(dict(m,id=i,workflow=by_id[i]["inputs"]["workflow"],split=next(s for s,v in splits.items() if i in v)))
    mask=np.asarray(arrays["option_mask"])
    alignerr=float(np.max(np.abs(expected[mask]-arrays["probabilities"][mask])))
    assert alignerr < 1e-4, (passno,alignerr)
    z,x,a=scorer_features(arrays["head_inputs"])
    out={"rows":rows,"gold":torch.tensor(gold),"mask":torch.tensor(mask),
         "logits":torch.tensor(np.nan_to_num(np.array(arrays["option_logits"]),nan=0)),
         "prob":torch.tensor(np.nan_to_num(np.array(arrays["probabilities"]),nan=0)),
         "z":torch.nan_to_num(z),"x":torch.nan_to_num(x)}
    print("Read",passno,"aligned",len(rows),"questions; max probability error",f"{alignerr:.2g}")
    return out
ds_parts=[load_read_dataset(k,g) for k,g in zip([2,3,4],read_groups[:3])]
json.dump({"type":"read","model":"original","from":{"pass":1,"ids":read_groups[3]}},
          open(WORK/"read_dev2.json","w"))
note("Read alignment verified by reproducing recorded probabilities; from-filter retains baseline record order, not supplied ID order.\n"
     "Next pass: READ original remaining 72 dev examples (360 questions), IDs in read_dev2.json. Exact edits: none. All 448 fit and 224 dev will have scorer activations; 228 test examples remain unseen for edit design.")
```
