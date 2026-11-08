The cached features support a useful correction: a closed-form scalar edit raises development accuracy from 37.3% to about 54.5%, while removing set attention alone does not help. I’ll now reproduce the set-attention computation exactly so that strength tests include its response to the changed scalar logits.
```python
# Hypothesis: the scalar readout is the main defect; exact set-head reconstruction will separate its effect from attention side effects.
import itertools
choice_idx=np.where(types=="choice")[0][:64]
H=torch.from_numpy(np.nan_to_num(A["head_inputs"][choice_idx])).float()
P=torch.from_numpy(np.nan_to_num(A["path_ends"][choice_idx])).float()
target_delta=torch.from_numpy(np.nan_to_num(A["set_delta"][choice_idx])).float()
cmask=torch.from_numpy(mask[choice_idx])
base_logits=torch.nn.functional.linear(H,head0["scalar.weight"],head0["scalar.bias"])
def set_compute_small(features,scalar,activation,heads,residual=False):
    u=torch.nn.functional.linear(torch.cat([features,scalar],dim=-1),head0["set_project.weight"],head0["set_project.bias"])
    u=activation(u)
    q,k,v=torch.nn.functional.linear(u,head0["set_attention.in_proj_weight"],head0["set_attention.in_proj_bias"]).chunk(3,dim=-1)
    B,O,E=q.shape; d=E//heads
    q,k,v=[t.view(B,O,heads,d).transpose(1,2) for t in [q,k,v]]
    scores=(q@k.transpose(-2,-1))/math.sqrt(d)
    scores=scores.masked_fill(~cmask[:,None,None,:],-1e20)
    out=(scores.softmax(dim=-1)@v).transpose(1,2).reshape(B,O,E)
    out=torch.nn.functional.linear(out,head0["set_attention.out_proj.weight"],head0["set_attention.out_proj.bias"])
    if residual: out=out+u
    return torch.nn.functional.linear(out,head0["set_output.weight"],head0["set_output.bias"]).squeeze(-1)
activations={"identity":lambda z:z,"tanh":torch.tanh,"gelu":torch.nn.functional.gelu,"relu":torch.relu,"silu":torch.nn.functional.silu}
matches=[]
for fkey,fv in [("head",H),("ends",P)]:
    for actkey,act in activations.items():
        for nh in [1,2,4,8,16]:
            for residual in [False,True]:
                pred=set_compute_small(fv,base_logits,act,nh,residual)
                diff=(pred-target_delta)[cmask]
                matches.append((float(diff.square().mean().sqrt()),float(diff.abs().max()),fkey,actkey,nh,residual))
print("Best formula matches:",sorted(matches)[:12])
# Distinguish per-question priors from state-dependent evidence in the fitted correction.
groupkeys=np.array([m["workflow"]+"/"+m["question"] for m in rows])
delta=solutions[0.01]
dc=X@delta
print("Fit correction norm",np.linalg.norm(delta),"cosine with original scalar",
      float(np.dot(delta,head0["scalar.weight"].numpy().ravel())/(np.linalg.norm(delta)*head0["scalar.weight"].norm().item())))
for key in sorted(set(groupkeys)):
    ix=(groupkeys==key)&dev_rows
    baseline_acc=metric_logits(L0,ix)[0]; new_acc=metric_logits(L0+dc,ix)[0]
    # Correction ranges across states for each option, restricted to this question schema.
    n=mask[ix][0].sum()
    print(key, "n",int(ix.sum()),"acc",round(baseline_acc,3),"->",round(new_acc,3),
          "correction_state_SD",np.std(dc[ix,:n],axis=0).round(2).tolist())
```
