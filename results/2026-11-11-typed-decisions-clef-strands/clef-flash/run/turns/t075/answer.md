The half-strength run provides the expected graded intervention. I’ll now restore the selected full repair and test robustness to irrelevant bookkeeping metadata. This checks whether the repaired readout follows business evidence rather than incidental record identifiers.

```python
# Hypothesis: best_2's evidence-sensitive gains should survive irrelevant metadata changes while the backbone and schema remain unchanged.
mm19,ff19,mk19,ll19=read_probe_arrays(19)
assert mm19==target_mm
print("Half-dose feature error:",float((ff19-target_ff).abs().max()))
print("Half-dose logit reconstruction error:",float((ll19-half_probe_logits)[target_mk].abs().max()))
print("Observed half-dose midpoint error:",float((ll19-.5*(ll17+ll18))[target_mk].abs().max()))
for name,arr in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],repair_b2["tensors"]):
    save_edit(name,arr.reshape(SHAPES[name]))
nuisance_requests=[]
nuisance_meta=[]
nuisance_seeds=[]
for wn in repair_workflows:
    nuisance_seeds.extend([ii for ii in splits["test"] if byid[ii]["inputs"]["workflow"]==wn][:10])
for si,ii in enumerate(nuisance_seeds):
    rq0=json.loads(basebyid[ii]["calls"][0]["prompt"])
    for level in ["original","short_metadata","long_metadata"]:
        rq=copy.deepcopy(rq0)
        if level=="short_metadata":
            rq["state"]["inputs"]["state"]["archive_metadata"]={"record_reference":f"ARCH-{80000+si*71}","export_version":"2"}
        elif level=="long_metadata":
            rq["state"]["inputs"]["state"]["archive_metadata"]={
                "record_reference":f"ARCH-{30000+si*93}",
                "export_version":"2",
                "archive_file":f"records/shard-{si%7}/batch-{si+11}.jsonl",
                "checksum":hashlib.sha256(str(ii).encode()).hexdigest(),
                "description":"Bookkeeping metadata identifying the archived copy; the business record above is unchanged."
            }
        nuisance_requests.append(rq)
        nuisance_meta.append({"seed":ii,"workflow":byid[ii]["inputs"]["workflow"],"level":level})
for model in ["original","edited"]:
    json.dump({"type":"read","model":model,"requests":nuisance_requests},open(WORK/f"nuisance_probes_{model}.json","w"))
json.dump(nuisance_meta,open(WORK/"nuisance_probe_meta.json","w"))
print("Nuisance probes:",len(nuisance_requests),"requests,",5*len(nuisance_requests),"questions")
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 19 dose verification\nHalf-extension actual logit margins match midpoint of shared-only and full extension to float32 precision; upstream features identical. This confirms graded causal control through the selected22 output entries. Full best_2 restored; no new candidate selected.\n")
    f.write("\n## Planned pass 20 — nuisance robustness original\nExact current edits best_2 unchanged. READ original on120 requests:40 former-test seeds(10/workflow), each original/short archive metadata/long archive metadata. Business evidence and schema unchanged. nuisance_probes_original.json. No gold supplied and no edit fitting; robustness diagnostic only.\n")
```
