Validation confirms the repair: 34.8% → 56.0%, a paired gain of 21.2 ± 1.6 percentage points. I’ll preserve that checkpoint, then distinguish how much of the gain comes from corrected option priors versus reading record-specific evidence.
```python
# Hypothesis: the validated correction contains both template-prior repair and a separate state-evidence readout.
import shutil
note("Forward8 hidden validation300 confirmed soft-full: original0.348000, edited0.560000, paired gain0.212000 SE0.015787;199 improved/42 worsened. Current best confirmed candidate: soft-full.")
best1=WORK/"best_1"
if best1.exists(): shutil.rmtree(best1)
shutil.copytree(WORK/"edited",best1)
torch.save({"scalar.weight":load("scalar.weight").clone()},WORK/"best_1_weights.pt")
# Option-template subspace from fitting examples only.
mean_rows=[]
for key in sorted(set(groupkeys)):
    ix=(groupkeys==key)&fit_rows
    n=int(mask[ix][0].sum())
    mean_rows.extend(Xc[ix,:n].mean(axis=0))
M=torch.tensor(np.array(mean_rows),dtype=torch.float64)
_,sing,Vh=torch.linalg.svd(M,full_matrices=False)
rank=int((sing>1e-6).sum())
Q=Vh[:rank].T.numpy()
d_prior=Q@(Q.T@d_soft)
d_evidence=d_soft-d_prior
print("Template-subspace rank",rank,"prior norm",np.linalg.norm(d_prior),"evidence norm",np.linalg.norm(d_evidence))
for name,dw in [("none",d_soft*0),("prior-only",d_prior),("evidence-only",d_evidence),("full",d_soft)]:
    logits=head_logits(w0+dw)
    print(name,"fit",metric_logits(logits,fit_rows),"dev",metric_logits(logits,dev_rows))
print("Soft vs mixed development accuracy by question:")
soft_pred=head_logits(w0+d_soft)
mix_pred=head_logits(w0+.75*d_mix)
for key in sorted(set(groupkeys)):
    ix=(groupkeys==key)&dev_rows
    print(key,round(metric_logits(L0,ix)[0],3),round(metric_logits(soft_pred,ix)[0],3),round(metric_logits(mix_pred,ix)[0],3))
# Diagnose the scalar feature entering set_project using a linearized inversion of this small head only.
zero=torch.zeros_like(base_logits)
delta_zero=set_formula(H,zero,activations["identity"],4,"project",activations["tanh"])
J=[]
for o in range(H.shape[1]):
    ss=zero.clone();ss[:,o,0]=1
    J.append(set_formula(H,ss,activations["identity"],4,"project",activations["tanh"])-delta_zero)
jac=torch.stack(J,dim=-1)
inferred=[]
for b in range(len(H)):
    n=int(cmask[b].sum())
    inferred.append(torch.linalg.lstsq(jac[b,:n,:n].double(),(target_delta[b,:n]-delta_zero[b,:n]).double()).solution.numpy())
print("Inferred set scalar inputs vs raw for first three choices:")
for b in range(3):
    n=int(cmask[b].sum())
    print("raw",base_logits[b,:n,0].numpy().round(4),"inferred",np.round(inferred[b],4))
note(f"Soft direction decomposition, fit-only mean option-template span rank{rank}: prior norm{np.linalg.norm(d_prior):.4f}, evidence norm{np.linalg.norm(d_evidence):.4f}. Full candidate backed up to best_1 and best_1_weights.pt. Next controlled prior-only ablation and matched-state cue tests.")
# Save a causally interpretable prior-only ablation for one development pass.
torch.save(torch.tensor(d_prior,dtype=torch.float32),candidate_dir/"prior_direction.pt")
torch.save(torch.tensor(d_evidence,dtype=torch.float32),candidate_dir/"evidence_direction.pt")
set_scalar_direction(d_prior,1)
note("Planned forward9: paired DECIDE dev150, same dev_decide_request.json. Exact edit scalar.weight=w_original+d_prior (projection of validated d_soft onto fit mean option-template span); all others original. Tests whether full gain can be explained by question/option priors alone. Restore best1 afterward.")
```
