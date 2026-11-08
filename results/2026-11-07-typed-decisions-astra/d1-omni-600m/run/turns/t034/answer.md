The over-strength test confirms the predicted decline. I’ll restore full strength and add the conditional extension only for questions the original model handled poorly. The extension uses weakly contributing scorer neurons and gates derived from question/option identity—not gold answers—so state-dependent corrections apply to the intended options.
```python
# Hypothesis: option-gated residual directions restore common state cues while leaving well-handled questions and other options largely unchanged.
note("Pass 11 over-strength 1.5x: dev 62.679%, +7.679pp SE1.866pp, exactly predicted and below 1x 67.054%. Confirms non-monotonic dose response; reverting to 1x for extension.")
selected_gate_keys=[(qno,o) for qno,qkey in enumerate(qkeys) if qkey in target_qs for o in range(int(D["n"][qid==qno][0]))]
# Gate thresholds are determined on fit identity labels only.
gate_thresholds=[]
for gn in range(len(qopt_keys)):
    pos=fitmask&(gate_actual==gn)
    neg=fitmask&(gate_actual!=gn)
    lower=float(torch.quantile(idgates[:,:,gn][pos],.01))
    upper=float(torch.quantile(idgates[:,:,gn][neg],.999))
    gate_thresholds.append((lower+upper)/2)
gate_thresholds=torch.tensor(gate_thresholds)
centered_idgates=idgates-gate_thresholds
sgidx=[qopt_index[q,o] for q,o in selected_gate_keys]
valid_selected=valid
print("Selected conditional gates:",len(sgidx),"neurons needed",2*len(sgidx))
print("Fit-derived thresholds range",float(gate_thresholds.min()),float(gate_thresholds.max()))
vg=centered_idgates[:,:,sgidx]
truth=ident[:,:,sgidx]>0
print("Selected gates dev classification errors",int(((vg>0)!=truth)[vld].sum()),
      "positive min",float(vg[vld][truth[vld]].min()),"negative max",float(vg[vld][~truth[vld]].max()))
# Build the actual scorer, not just an oracle-gated prediction.
cond_units=[int(v) for v in neuron_order[2:2+2*len(sgidx)]]
def make_conditional(strength=1.0,gate_scale=30.,epsilon=.025,rel=.1):
    w1=scorer_params["1.weight"].clone();b1=scorer_params["1.bias"].clone();w3=scorer_params["3.weight"].clone()
    j,k=pair_units
    a=candidate_dx/candidate_dx.norm(); bias=-float(a@mu_x)
    w1[j]=a;w1[k]=-a;b1[j]=bias;b1[k]=-bias
    w3[0,j]=candidate_dx.norm();w3[0,k]=-candidate_dx.norm()
    for it,(qno,o) in enumerate(selected_gate_keys):
        gn=qopt_index[qno,o]
        gw=gate_scale*idgateW[:,gn]
        gb=gate_scale*float(idmean[gn]-mu_x@idgateW[:,gn]-gate_thresholds[gn])
        d,b=conditional_solutions[qkeys[qno]][o][rel]
        j,k=cond_units[2*it:2*it+2]
        w1[j]=gw+epsilon*d;w1[k]=gw-epsilon*d
        b1[j]=gb+epsilon*b;b1[k]=gb-epsilon*b
        w3[0,j]=strength/(2*epsilon);w3[0,k]=-strength/(2*epsilon)
    return w1,b1,w3
def logits_scorer_params(x,params):
    w1,b1,w3=params
    return F.linear(F.gelu(F.linear(x,w1,b1)),w3,scorer_params["3.bias"]).squeeze(-1)
cond_params=make_conditional()
cond_logits=logits_scorer_params(D["x"],cond_params)
oracle=base_linear+selected_add
print("Conditional actual local dev:",evaluate_logits(cond_logits))
print("Oracle deviation quantiles:",torch.quantile((centered(cond_logits)-centered(oracle)).abs()[D["dev"][:,None]&D["mask"]],torch.tensor([.5,.9,.99,1.])).tolist())
for st in [0,0.5,1,1.5,2]:
    l=logits_scorer_params(D["x"],make_conditional(strength=st))
    print("Actual conditional dose",st,{k:round(v,5) for k,v in evaluate_logits(l).items()})
def save_scorer_params(params):
    for n,v in zip(["head.scorer.1.weight","head.scorer.1.bias","head.scorer.3.weight"],params):save_edit(n,v)
save_scorer_params(cond_params)
torch.save({"params":cond_params,"question_options":selected_gate_keys,"units":cond_units,
            "strength":1.,"gate_scale":30.,"epsilon":.025,"ridge":.1},WORK/"conditional_candidate.pt")
note("Next pass: DECIDE both 224 dev IDs in dose_dev.json. Exact current edits: full-strength base residual pair33/746 plus "+str(len(sgidx))+
     " option-gated residual channels using "+str(len(cond_units))+
     " low-contribution scorer neurons (indices in conditional_candidate.pt). Gates fit on q/option identities only; target questions selected by original fit accuracy <.60, residual ridge=.1; strength1, gate scale30, epsilon.025. Parameter copies conditional_candidate.pt. Local predicted dev result "+str(evaluate_logits(cond_logits))+".")
```
This candidate is ready for a paired development test; the validated two-neuron repair remains backed up in `best_1`.
