I’ll compare the remaining errors by question and inspect a few security failures. This will help distinguish a specific missed cue from a broad calibration problem; I won’t change weights in this step.


```python
# Hypothesis: security credential false positives and invoice duplicate misses have identifiable state cues that the repaired scorer still underuses.
print("D fields:", [(k,tuple(v.shape) if isinstance(v,torch.Tensor) else (type(v).__name__,len(v) if hasattr(v,"__len__") else "")) for k,v in D.items()])
print("Diagnostic candidate value structures:",
      {str(k):(type(v).__name__,[(type(x).__name__,tuple(x.shape) if isinstance(x,torch.Tensor) else len(x) if hasattr(x,"__len__") else "") for x in v] if isinstance(v,tuple) else list(v)[:5] if isinstance(v,dict) else "") for k,v in list(conditional_candidates.items())[:1]})
print("qdata keys / example:",qkeys[:2], {k:(tuple(v.shape) if isinstance(v,torch.Tensor) else str(v)[:100]) for k,v in next(iter(qdata.values())).items()})
print("Existing table first 6:",conditional_table[:6])
print("Signatures:", {fn: str(inspect.signature(globals()[fn])) for fn in ["summary_logits","center_options","pair_tensors","edited_pair_logits","targeted_tensors","simulate_scoring_tensors"]})
def unpack_probs(out):
    if isinstance(out,dict) and "probabilities" in out: return out["probabilities"]
    return out
print("One output",r10[0]["outputs"])
pgroups = collections.defaultdict(lambda: [0,0,0,0])
for rt in r10:
    bo = basebyid[rt["id"]]
    b1 = r5by[rt["id"]]
    ex = byid[rt["id"]]
    for qid,gg in ex["labels"]["gold"].items():
        truth=max(gg["probabilities"],key=gg["probabilities"].get)
        vv=pgroups[(ex["inputs"]["workflow"],qid)]
        for nrec,rec in enumerate([bo,b1,rt]):
            po=unpack_probs(rec["outputs"][qid])
            if isinstance(po,dict):
                vv[nrec]+= (max(po,key=po.get)==truth)
        vv[3]+=1
print("Reserved-practice accuracy by field (original,best1,best2):")
for k,v in sorted(pgroups.items()):
    print(k[0][:8],k[1], [round(100*a/v[3],1) for a in v[:3]])
sec_ids = [ii for ii in splits["fit"] if byid[ii]["inputs"]["workflow"]=="security_incidents"]
for ii in sec_ids:
    ex=byid[ii]; gg=ex["labels"]["gold"]["credential_compromise"]["probabilities"]
    po=unpack_probs(basebyid[ii]["outputs"]["credential_compromise"])
    if max(gg,key=gg.get) != max(po,key=po.get):
        print("Security miss:",ii, "state",ex["inputs"]["state"],"gold",gg,"original",po)
        if sum(1 for x in sec_ids[:sec_ids.index(ii)+1] if max(byid[x]["labels"]["gold"]["credential_compromise"]["probabilities"], key=byid[x]["labels"]["gold"]["credential_compromise"]["probabilities"].get) != max(unpack_probs(basebyid[x]["outputs"]["credential_compromise"]),key=unpack_probs(basebyid[x]["outputs"]["credential_compromise"]).get))>=4: break
```
