Validation confirms the edit: accuracy rises from 70.20% to 76.27%, a gain of 6.07 points with paired SE 0.81 points. I’ll preserve it as the current best. Next I’ll examine which errors it leaves behind, especially whether one shared correction interferes across different question types.


```python
# Hypothesis: the validated shared bypass helps, but different questions still require distinct state-to-option readout directions.
best1 = WORK/"best_1"
if best1.exists():
    shutil.rmtree(best1)
shutil.copytree(WORK/"edited",best1)
note("## Pass 6 validation — confirmed best_1\nn=300, original 0.702000, edited 0.762667; gain +0.060667, paired SE 0.008052; 97 improved, 20 worsened. Current best: two-neuron bypass (791,1015), saved in best_1/. This is the fallback for all further experiments.")
tuneL = edited_pair_logits(.5)
basepred = D["logits"].masked_fill(~D["mask"],-1e9).argmax(-1)
newpred = tuneL.masked_fill(~D["mask"],-1e9).argmax(-1)
print("Tune field performance, baseline -> confirmed bypass:")
qkeys = sorted(groups.keys())
D["qkey"]=[(byid[i]["inputs"]["workflow"],m["question"]) for i,m in zip(D["ids"],D["meta"])]
for wf,q in qkeys:
    ix=torch.tensor([k==(wf,q) for k in D["qkey"]]) & D["istune"]
    bc=(basepred[ix]==D["target"][ix]).float().mean().item()
    ec=(newpred[ix]==D["target"][ix]).float().mean().item()
    print(wf[:9],q,f"{bc:.2f}->{ec:.2f}")
# Determine which feature blocks actually carry the successful correction.
for label,sl in [("option",slice(0,1024)),("field*option",slice(1024,2048)),("|field-option|",slice(2048,3072))]:
    dpart=torch.zeros_like(bypass_delta)
    dpart[sl]=bypass_delta[sl]
    L=D["logits"]+.5*(D["FC"]@dpart)
    print("Block alone",label,summary_logits(L,D["istune"]))
print("Feature block correction norms:",[bypass_delta[i*1024:(i+1)*1024].norm().item() for i in range(3)])
```
