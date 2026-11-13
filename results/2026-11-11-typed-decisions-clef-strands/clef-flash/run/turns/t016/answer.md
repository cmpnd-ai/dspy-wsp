The best direct-input correction improves tuning accuracy by 5.6 points. I’ll test it as a small bypass around the compressed scorer layer: two nearly inactive neurons with opposite inputs and outputs. Since GELU(z) − GELU(−z) = z, they can carry the missing linear direction without replacing the existing scorer.


I’ll simulate bfloat16 rounding and several strengths before saving anything.

```python
# Hypothesis: adding a cue-sensitive path through two inactive GELU neurons causally restores distinctions lost by residual_scorer.0.
bypass_delta = alt_candidates["hard_margin",1e-4].clone()
pair = least[:2].tolist()
pair_gain = torch.tensor(1/gate).to(torch.bfloat16).float().item()
def pair_tensors(strength, delta=bypass_delta, neurons=pair):
    row = torch.cat([torch.zeros(1024), strength*delta])
    wfirst = W0.clone()
    bfirst = b0.clone()
    wlast = wout.clone()
    for neuron,sign in zip(neurons,[1.,-1.]):
        wfirst[neuron] = sign*row
        bfirst[neuron] = 0.
        wlast[neuron] = sign*pair_gain
    return (wfirst.to(torch.bfloat16).float(),
            bfirst.to(torch.bfloat16).float(),
            wlast.to(torch.bfloat16).float())
def edited_pair_logits(strength):
    wf,bf,wl=pair_tensors(strength)
    old = D["activations"][:,:,pair]@wout[pair]
    new = F.gelu(F.linear(D["features"],wf[pair],bf[pair]))@wl[pair]
    return D["logits"]+gate*(new-old)

print("Pair:",pair,"gain:",pair_gain,"direction norm:",bypass_delta.norm().item())
dose_results = []
for strength in [0.,.125,.25,.5,.75,1.]:
    L = edited_pair_logits(strength)
    t = summary_logits(L,D["istune"])
    f = summary_logits(L,D["isfit"])
    dose_results.append((strength,f,t))
    print(strength,"fit",round(f["accuracy"],4),"tune", {k:round(float(v),5) for k,v in t.items()})
Lbest=edited_pair_logits(.5)
effect = center_options(Lbest-D["logits"],D["mask"])
print("Predicted corrections to representative fit failures:")
for exid,q in [("train_0","outcome"),("train_8","outcome"),("train_3","urgency"),("train_901","credential_compromise"),("train_663","duplicate")]:
    ii=[i for i,(eid,m) in enumerate(zip(D["ids"],D["meta"])) if eid==exid and m["question"]==q]
    if ii:
        i=ii[0]; opts=D["meta"][i]["options"]
        print(exid,q,"target",opts[D["target"][i]],"corrections",dict(zip(opts,effect[i,:len(opts)].numpy().round(3).tolist())),
              "before/after",opts[D["logits"][i,:len(opts)].argmax()],opts[Lbest[i,:len(opts)].argmax()])
wf,bf,wl=pair_tensors(.5)
for name,val in [("residual_scorer.0.weight",wf),("residual_scorer.0.bias",bf),("residual_scorer.3.weight",wl[None,:])]:
    save_edit(name,val)
torch.save({"delta":bypass_delta,"pair":pair,"strength":.5,"first_weight":wf,"first_bias":bf,"last_weight":wl},WORK/"linear_bypass_candidate.pt")
json.dump({"type":"decide","model":"both","problems":splits["test"]},(WORK/"test_linear_bypass.json").open("w"))
note("## Causal bypass experiment\nRepurposed scorer neurons "+str(pair)+" (zero within-option contribution at baseline), with opposite incoming rows and opposite outgoing weights. GELU(z)-GELU(-z)=z, so this adds the closed-form missing direction while preserving original scorer. All tensors rounded to bfloat16 before simulation. Dose response: "+str([(s,round(t["accuracy"],4)) for s,f,t in dose_results])+".\n## Planned pass 5\nExact edit: residual_scorer.0.weight rows "+str(pair)+", residual_scorer.0.bias same entries set zero, residual_scorer.3.weight same entries ±"+str(pair_gain)+". Direction: hard-margin ridge 1e-4, strength 0.5, fit 500 only; saved linear_bypass_candidate.pt. DECIDE both on all 300 reserved test IDs, test_linear_bypass.json. No test labels used for selection.")
```
