The successful edit shows that useful evidence survives into the final features but is not fully read by the original scorer. I’ll test a narrower hypothesis next: one shared correction may mix incompatible decision boundaries across the four workflows. A workflow-specific linear correction could address that without changing attention or the backbone.


```python
# Hypothesis: the shared final-scoring direction mixes distinct workflow boundaries; workflow-conditional directions may generalize better.
pgroups=collections.defaultdict(lambda:[0,0,0,0])
for rt in r10:
    records=[basebyid[rt["id"]],r5by[rt["id"]],rt]
    answers=[call_answers(rec) for rec in records]
    ex=byid[rt["id"]]
    for qid,gg in ex["labels"]["gold"].items():
        truth=max(gg["probabilities"],key=gg["probabilities"].get)
        vv=pgroups[(ex["inputs"]["workflow"],qid)]
        for j,an in enumerate(answers):
            po=an[qid]["probabilities"]
            vv[j]+=max(po,key=po.get)==truth
        vv[3]+=1
print("Reserved test descriptive results (NOT used to select the new direction):")
for k,v in sorted(pgroups.items()):
    print(k[0][:8],k[1],[round(100*a/v[3],1) for a in v[:3]])
fit_sec_errors=[ii for ii in splits["fit"] if byid[ii]["inputs"]["workflow"]=="security_incidents" and
                max(call_answers(basebyid[ii])["credential_compromise"]["probabilities"],key=call_answers(basebyid[ii])["credential_compromise"]["probabilities"].get)
                != max(byid[ii]["labels"]["gold"]["credential_compromise"]["probabilities"],key=byid[ii]["labels"]["gold"]["credential_compromise"]["probabilities"].get)]
for ii in fit_sec_errors[:3]:
    print("Security miss:",ii,byid[ii]["inputs"]["state"],
          "gold",byid[ii]["labels"]["gold"]["credential_compromise"]["probabilities"],
          "original",call_answers(basebyid[ii])["credential_compromise"]["probabilities"])
print("Feature stats:",float(D["features"].nan_to_num().square().mean().sqrt()),
      "FC std",float(D["FC"].square().mean().sqrt()),"shared correction rms",float(shared_correction[D["mask"]].square().mean().sqrt()))
print("Target stats:", [(k,float(v[D["mask"]].min()),float(v[D["mask"]].max())) for k,v in alt_targets.items()])
print("Tensor save formats:",[(p.name,type(torch.load(p,weights_only=False)).__name__) for p in [WORK/"linear_bypass_candidate.pt",WORK/"targeted_candidate.pt"]])
print("By field tune, original / shared / targeted:")
for qk in qkeys:
    ixq=torch.tensor([z==qk for z in D["qkey"]]) & D["istune"]
    print(qk, [round(float(((z.argmax(-1)==D["target"])[ixq]).float().mean()),3) for z in [D["logits"],Lbest,Ltarget]])
with open(WORK/"notes.md","a") as f:
    f.write("\n## Next hypothesis: workflow-conditional missing scoring directions\nShared correction helped substantially; question-specific extension adds uncertain benefit. Will test one-shot workflow ridge corrections on original option-centered features, fit 500 only, tune 100. No test labels used in fitting/selection. Workflow gates will use only schema identities, not answers. best_2 remains saved.\n")
```
