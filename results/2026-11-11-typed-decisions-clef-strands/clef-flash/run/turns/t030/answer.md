The covariance-aware identity read separates the targeted options on every tuning record. I’ll use it to simulate narrowly gated corrections for agent urgency and customer handling. Each correction will activate only for its intended question and option; I’ll measure leakage and bfloat16 effects before saving it.

```python
# Hypothesis: gated field-evidence reads repair three identified failure fields while preserving the validated bypass elsewhere.
gcoef,gbias,gscores=gate_models[1e-5]
selected_classes=[c for c,(q,o) in enumerate(option_classes) if q in target_qkeys]
new_neurons=least[2:2+2*len(selected_classes)].tolist()
base_wf,base_bf,base_wl=pair_tensors(.5)
actual_shared_row=base_wf[pair[0]]*(gate*base_wl[pair[0]])
rawshared=D["features"]@actual_shared_row
qoffset={}
for qkey in target_qkeys:
    ix=torch.tensor([q==qkey for q in D["qkey"]])&D["isfit"]
    qoffset[qkey]=rawshared[ix][D["mask"][ix]].mean().item()
expert_rows={}
for c in selected_classes:
    qkey,opt_id=option_classes[c]
    rowi=D["qkey"].index(qkey)
    j=D["meta"][rowi]["options"].index(opt_id)
    coef,intercept=conditional_candidates["soft",.01]["param"][qkey]
    r=-actual_shared_row.clone()
    r[:1024]+=coef[:,j]
    b=intercept[j].item()+qoffset[qkey]
    expert_rows[c]=(r,b)
def targeted_tensors(gate_strength=32.,expert_strength=1.):
    wf,bf,wl=base_wf.clone(),base_bf.clone(),base_wl.clone()
    for k,c in enumerate(selected_classes):
        r,b=expert_rows[c]
        gv=torch.cat([gcoef[:,c],torch.zeros(2048)])*gate_strength
        gb=(gbias[c]-.5)*gate_strength
        for sign,neuron in zip([1.,-1.],new_neurons[2*k:2*k+2]):
            wf[neuron]=gv+sign*.5*expert_strength*r
            bf[neuron]=gb+sign*.5*expert_strength*b
            wl[neuron]=sign*pair_gain
    return wf.to(torch.bfloat16).float(),bf.to(torch.bfloat16).float(),wl.to(torch.bfloat16).float()
all_modified=pair+new_neurons
def simulate_scoring_tensors(tensors,neurons=all_modified):
    wf,bf,wl=tensors
    new=F.gelu(F.linear(D["features"],wf[neurons],bf[neurons]))@wl[neurons]
    old=D["activations"][:,:,neurons]@wout[neurons]
    return D["logits"]+gate*(new-old)
print("Gated extension uses",len(new_neurons),"additional neurons:",new_neurons)
for gs in [16.,32.,64.,128.]:
    for strength in [.5,1.]:
        L=simulate_scoring_tensors(targeted_tensors(gs,strength))
        t=summary_logits(L,D["istune"])
        leak=center_options(L-edited_pair_logits(.5),D["mask"])[(~targetmask)&D["istune"]]
        print("gate",gs,"strength",strength,"fit",round(summary_logits(L,D["isfit"])["accuracy"],4),
              "tune",round(t["accuracy"],4),"gain/SE",round(t["gain"],4),round(t["se"],4),
              "non-target leak RMS/max",round(leak.square().mean().sqrt().item(),4),round(leak.abs().max().item(),4))
note("Covariance-aware identity readout: fit/tune option identity accuracy 100% at ridge 1e-5; targeted tune class scores min positive 0.651, max negative 0.411, threshold 0.5. Simulating 22 additional weak-neuron replacements with symmetric gated GELU differences. Each correction is a field-evidence direction minus the shared correction, gated to exactly one question-option identity.")
```
