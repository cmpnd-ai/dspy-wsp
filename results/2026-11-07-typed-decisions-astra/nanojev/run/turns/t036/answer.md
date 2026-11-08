I’ll verify the set-head formula against the read outputs, then test whether its 128 nonlinear features carry residual choice evidence. Any proposed set-output edit will be derived on the original 450 fitting records and assessed on the 150 development records.
```python
# Hypothesis: nonlinear option-set features carry residual choice evidence not captured by the repaired scalar projection.
meta14,a14=plain_read(14)
row_lookup={(m["id"],m["question"]):j for j,m in enumerate(rows)}
i14=np.array([row_lookup[(choice600_manifest[m["request"]],m["question"])] for m in meta14["rows"]])
print("Choice read vs previous head features max",np.nanmax(np.abs(a14["head_inputs"]-A["head_inputs"][i14])))
print("Choice set correction invariant max",np.nanmax(np.abs(a14["set_delta"]-A["set_delta"][i14])))
def get_set_features():
    Bset=torch.zeros((len(rows),mask.shape[1],128),dtype=torch.float32)
    for start in range(0,len(choice_all),128):
        ix=choice_all[start:start+128];h=HX[ix]
        c=MX[ix].sum(1).float().log()[:,None,None].expand(-1,HX.shape[1],1)
        u=torch.nn.functional.linear(torch.cat([h,c],-1),head0["set_project.weight"],head0["set_project.bias"])
        q,k,v=torch.nn.functional.linear(u,head0["set_attention.in_proj_weight"],head0["set_attention.in_proj_bias"]).chunk(3,-1)
        nb,no,ne=q.shape
        q,k,v=[t.reshape(nb,no,4,32).transpose(1,2) for t in [q,k,v]]
        scores=((q@k.transpose(-2,-1))/math.sqrt(32)).masked_fill(~MX[ix,None,None,:],-1e20)
        att=(scores.softmax(-1)@v).transpose(1,2).reshape(nb,no,ne)
        att=torch.nn.functional.linear(att,head0["set_attention.out_proj.weight"],head0["set_attention.out_proj.bias"])
        Bset[ix]=torch.tanh(u+att)
    return Bset.numpy().astype(np.float64)
Bset=get_set_features()
set_rec=Bset@head0["set_output.weight"].numpy().reshape(-1)+head0["set_output.bias"].item()
ci_mask=mask & (types=="choice")[:,None]
print("Exact set formula max error:",np.max(np.abs(set_rec-A["set_delta"])[ci_mask]))
Bc=(Bset-(Bset*mask[...,None]).sum(1,keepdims=True)/counts[:,None,None])*mask[...,None]
DDset=(Bc[fit_rows]*RW[fit_rows,None,None])[mask[fit_rows]]
DDsett=torch.from_numpy(DDset)
gset=DDsett.T@DDsett
evset,eset=torch.linalg.eigh(gset)
base450=L0+X@(.75*d_mix)
# Keep the same effective target as .75-strength scalar, so this is a residual contrast test.
effective_target=L0c+.75*(Tmix-L0c)
residual450=effective_target-centered(base450)
rset=(residual450[fit_rows]*RW[fit_rows,None])[mask[fit_rows]]
pset=eset.T@(DDsett.T@torch.from_numpy(rset))
set_candidates={}
print("Set-only residual correction over mixed450 baseline:",metric_logits(base450,dev_rows))
for lam in [1e-6,1e-5,.0001,.001,.01,.1,1]:
    ds=(eset@(pset/(evset+lam))).numpy()
    set_candidates[lam]=ds
    pp=base450+Bset@ds
    print("set-ridge",lam,"norm",np.linalg.norm(ds),"fit",metric_logits(pp,fit_rows),"dev",metric_logits(pp,dev_rows))
# Joint closed-form projection, with a separate regularizer for nonlinear set-output features.
Jt=torch.cat([Dt,DDsett],dim=1)
jg=Jt.T@Jt
jrhs=Jt.T@torch.from_numpy(((Tmix-L0c)[fit_rows]*RW[fit_rows,None])[mask[fit_rows]])
joint_solutions={}
for regset in [.00001,.0001,.001,.01,.1]:
    diag=torch.cat([torch.full((1024,),.003,dtype=torch.float64),torch.full((128,),regset,dtype=torch.float64)])
    sol=torch.linalg.solve(jg+torch.diag(diag),jrhs).numpy()
    joint_solutions[regset]=sol
    pred=L0+.75*(X@sol[:1024]+Bset@sol[1024:])
    print("joint ridge set",regset,"norms",np.linalg.norm(sol[:1024]),np.linalg.norm(sol[1024:]),
          "fit",metric_logits(pred,fit_rows),"dev",metric_logits(pred,dev_rows))
note("Forward14 verified scalar-independent set correction; exact set formula: tanh(project([head_input,log(option_count)]) + four-head self-attention output), then set_output. Testing fit450-only residual/joint closed-form set-output edits on dev150; no test or validation labels used.")
```
