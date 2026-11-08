Option-conditional readouts improve the development estimate, but I’ll check their gating and cue behavior before installing them. I’ll also run an over-strength version of the already validated channel: its predicted decline is a useful control against the idea that “more correction is always better.”
Option-conditional readouts improve the development estimate, but I’ll check their gates and cue behavior before installing them. I’ll also prepare an over-strength test of the validated channel; its predicted decline is a useful control against simply amplifying the correction indefinitely.
```python
# Hypothesis: conditional residual channels recover common state cues, while over-amplifying the shared channel should produce the predicted degradation.
truegate=idgates.gather(2,gate_actual[:,:,None]).squeeze(-1)
falsegate=idgates.clone()
falsegate.scatter_(2,gate_actual[:,:,None],-1e6)
print("Conditional gates dev min true",float(truegate[vld].min()),"max false",float(falsegate[vld].max()))
print("Weak neuron contrast RMS at ranks",[ (k,float(contrib_rms[neuron_order[k-1]])) for k in [2,20,40,80,120,160]])
# Restrict the extension to questions the original model demonstrably handles poorly on fit examples.
target_qs=[]
for qno,qkey in enumerate(qkeys):
    fitq=D["fit"]&(qid==qno)
    acc=float(D["correct0"][fitq].float().mean())
    if acc<.60:
        target_qs.append(qkey)
print("Baseline-error-targeted questions",target_qs)
selected_add=allcond[0.1].clone()
for j,r in enumerate(D["rows"]):
    if (r["workflow"],r["question"]) not in target_qs:selected_add[j]=0
for st in [0.25,0.5,0.75,1]:
    print("Conditional extension dose",st,{k:round(v,5) for k,v in evaluate_logits(base_linear+st*selected_add).items()})
cfcond=cflogits.clone()
for j,row in enumerate(cfmeta):
    tag=cf_tags[row["request"]]
    key=(by_id[tag["id"]]["inputs"]["workflow"],row["question"])
    if key in target_qs:
        for o in range(len(row["options"])):
            dd,bb=conditional_solutions[key][o][.1]
            cfcond[j,o]+=cfx[j,o]@dd+bb
cfvc=cf_values(cfcond)
for family,low,high,q in cfcontrasts:
    ids=sorted({t["id"] for t in cf_tags if t["family"]==family})
    old=np.mean([cfve[family,i,high,q]-cfve[family,i,low,q] for i in ids])
    new=np.mean([cfvc[family,i,high,q]-cfvc[family,i,low,q] for i in ids])
    print("Conditional predicted",family,q,round(old,4),"->",round(new,4))
install_channel(candidate_dx,1.5,"candidate_overstrong")
note("Next pass: DECIDE both same 224 dev IDs in dose_dev.json, shared channel strength 1.5, no conditional channels installed. Exact edits same scorer rows/biases33/746, output weights +/-1.5*||candidate_dx||. Preregistered prediction accuracy 62.679%, gain +7.679pp, paired SE 1.866pp; lower than full-strength 67.054%. Best_1 safely backed up.")
```
