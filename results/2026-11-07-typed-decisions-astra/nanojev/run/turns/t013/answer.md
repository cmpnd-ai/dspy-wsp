The correction varies with record content, not just question identity, and it is almost orthogonal to the original scalar weights. I’ll test simpler contrast targets and finish checking the set-head formula before committing to a weight edit.
```python
# Hypothesis: bounded gold-probability contrasts yield a more stable evidence readout than large log-probability contrasts.
# Extend only the permitted decision-head reconstruction, not the backbone.
def set_formula(features,scalar,preact,heads,residual,postact,logit_first=False):
    z=torch.cat([scalar,features],dim=-1) if logit_first else torch.cat([features,scalar],dim=-1)
    p=torch.nn.functional.linear(z,head0["set_project.weight"],head0["set_project.bias"])
    u=preact(p)
    q,k,v=torch.nn.functional.linear(u,head0["set_attention.in_proj_weight"],head0["set_attention.in_proj_bias"]).chunk(3,dim=-1)
    B,O,E=q.shape; d=E//heads
    q,k,v=[t.view(B,O,heads,d).transpose(1,2) for t in [q,k,v]]
    scores=((q@k.transpose(-2,-1))/math.sqrt(d)).masked_fill(~cmask[:,None,None,:],-1e20)
    out=(scores.softmax(dim=-1)@v).transpose(1,2).reshape(B,O,E)
    out=torch.nn.functional.linear(out,head0["set_attention.out_proj.weight"],head0["set_attention.out_proj.bias"])
    if residual=="project": out+=p
    if residual=="activated": out+=u
    out=postact(out)
    return torch.nn.functional.linear(out,head0["set_output.weight"],head0["set_output.bias"]).squeeze(-1)
matches2=[]
for fkey,fv in [("head",H),("ends",P)]:
    for pre in ["identity","tanh","gelu"]:
        for post in ["identity","tanh","gelu"]:
            for nh in [1,2,4,8,16]:
                for res in ["none","project","activated"]:
                    for first in [False,True]:
                        pr=set_formula(fv,base_logits,activations[pre],nh,res,activations[post],first)
                        diff=(pr-target_delta)[cmask]
                        matches2.append((float(diff.square().mean().sqrt()),float(diff.abs().max()),fkey,pre,nh,res,post,first))
print("Best set formula fits:",sorted(matches2)[:5])
# Closed-form alternatives share the same design and eigendecomposition; each is solved independently, without optimization.
def centered(a):
    return (a-(a*mask).sum(axis=1,keepdims=True)/counts[:,None])*mask
onehot=np.eye(mask.shape[1])[G.argmax(axis=-1)]
target_specs={
    "log01":Y,
    "prob3":centered(3*G),
    "prob5":centered(5*G),
    "mix3":centered(3*(0.5*G+0.5*onehot)),
    "hard2":centered(2*onehot),
}
closed_form={}
summary=[]
for target_name,T in target_specs.items():
    yr=((T-L0c)[fit_rows]*RW[fit_rows,None])[mask[fit_rows]]
    pr=eigvecs.T@(Dt.T@torch.from_numpy(yr))
    for lam in [.001,.003,.01,.03,.1,.3,1]:
        dw=(eigvecs@(pr/(eigvals+lam))).numpy()
        closed_form[(target_name,lam)]=dw
        logits=L0+X@dw
        f=metric_logits(logits,fit_rows);d=metric_logits(logits,dev_rows)
        summary.append((d[0],d[1],f[0],target_name,lam,float(np.linalg.norm(dw))))
print("Best contrast candidates (cached original set correction):")
for s in sorted(summary,reverse=True)[:16]: print(tuple(round(x,4) if isinstance(x,float) else x for x in s))
note("Diagnosis: no-set-attention dev accuracy 0.370667 vs original 0.373333, so set attention is not main defect. Scalar closed-form log-probability correction ridge0.01: fit0.638667/dev0.545333 (cached original set correction); norm100.35, cosine with original scalar0.0051. Corrections vary over records (~0.5–2 logit SD), including category and severity cues. Exploring bounded contrasts as an independently solved closed-form alternative, not iterative loss fitting.")
```
