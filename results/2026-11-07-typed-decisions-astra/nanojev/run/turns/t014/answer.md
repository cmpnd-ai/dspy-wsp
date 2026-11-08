Bounded probability contrasts generalize better than log-probability targets on development records. I’ll use the smaller, soft-label correction as the main candidate, check its dose-response in the reconstructed head, and save a half-strength intervention for the first causal test.
```python
# Hypothesis: strengthening the bounded-probability readout direction should progressively recover useful distinctions.
# Refine the set-head formula using original outputs only, not labels.
scalar_forms={
    "raw":base_logits,
    "zero":torch.zeros_like(base_logits),
    "sigmoid":base_logits.sigmoid(),
    "tanh":base_logits.tanh(),
    "centered":base_logits-(base_logits.squeeze(-1)*cmask).sum(dim=1)[:,None,None]/cmask.sum(dim=1)[:,None,None],
    "softmax":base_logits.squeeze(-1).masked_fill(~cmask,-1e20).softmax(dim=1).unsqueeze(-1),
}
formula_tests=[]
for sk,sv in scalar_forms.items():
    pr=set_formula(H,sv,activations["identity"],4,"project",activations["tanh"],False)
    diff=(pr-target_delta)[cmask]
    formula_tests.append((float(diff.square().mean().sqrt()),float(diff.abs().max()),sk))
print("Set scalar-input tests:",sorted(formula_tests))
# Vectorized exact-form head recomputation from read features (no backbone operations).
HX=torch.from_numpy(np.nan_to_num(A["head_inputs"])).float()
MX=torch.from_numpy(mask)
choice_all=np.where(types=="choice")[0]
def head_logits(weight, head=head0, batch=256):
    w=torch.as_tensor(weight,dtype=torch.float32).reshape(1,-1)
    z=torch.nn.functional.linear(HX,w,head["scalar.bias"]).squeeze(-1)
    z[torch.from_numpy(types=="noul"),0]=0
    for start in range(0,len(choice_all),batch):
        ix=choice_all[start:start+batch]; h=HX[ix]; base=z[ix].unsqueeze(-1)
        u=torch.nn.functional.linear(torch.cat([h,base],dim=-1),head["set_project.weight"],head["set_project.bias"])
        q,k,v=torch.nn.functional.linear(u,head["set_attention.in_proj_weight"],head["set_attention.in_proj_bias"]).chunk(3,dim=-1)
        B,O,E=q.shape; nh=4; d=E//nh
        q,k,v=[t.reshape(B,O,nh,d).transpose(1,2) for t in [q,k,v]]
        sc=((q@k.transpose(-2,-1))/math.sqrt(d)).masked_fill(~MX[ix,None,None,:],-1e20)
        attn=(sc.softmax(dim=-1)@v).transpose(1,2).reshape(B,O,E)
        attn=torch.nn.functional.linear(attn,head["set_attention.out_proj.weight"],head["set_attention.out_proj.bias"])
        corr=torch.nn.functional.linear(torch.tanh(u+attn),head["set_output.weight"],head["set_output.bias"]).squeeze(-1)
        z[ix]+=corr
    return z.numpy()
recon0=head_logits(head0["scalar.weight"])
print("All-row set-head reconstruction max / RMS:",
      np.max(np.abs((recon0-L0)[mask])),np.mean((recon0-L0)[mask]**2)**.5)
w0=head0["scalar.weight"].numpy().ravel().astype(np.float64)
d_soft=closed_form[("prob3",.01)]
d_mix=closed_form[("mix3",.003)]
for name,dw in [("soft",d_soft),("mix",d_mix)]:
    for strength in [0,.25,.5,.75,1,1.25,1.5]:
        pred=head_logits(w0+strength*dw)
        print(name,strength,"fit",tuple(round(v,4) for v in metric_logits(pred,fit_rows)),
              "dev",tuple(round(v,4) for v in metric_logits(pred,dev_rows)))
# Save first causal intervention at half-strength; no bias, norm, set head, or backbone changes.
candidate_dir=WORK/"candidate_weights"; candidate_dir.mkdir(exist_ok=True)
torch.save(torch.tensor(d_soft,dtype=torch.float32),candidate_dir/"soft_direction.pt")
torch.save(torch.tensor(d_mix,dtype=torch.float32),candidate_dir/"mix_direction.pt")
def set_scalar_direction(direction,strength):
    weight=torch.tensor(w0+strength*np.asarray(direction),dtype=torch.float32).reshape(1,1024)
    save_edit("scalar.weight",weight)
    return weight
w_half=set_scalar_direction(d_soft,.5)
print("Saved edit:",edited(),"relative angle cosine",float(torch.nn.functional.cosine_similarity(w_half,head0["scalar.weight"]).item()))
dev_decide={"type":"decide","model":"both","problems":splits["dev"]}
(WORK/"dev_decide_request.json").write_text(json.dumps(dev_decide))
note("Closed-form soft direction: fit450, centered targets 3*gold probability, ridge0.01, delta norm68.7146. Chosen as main candidate for lower norm and better Brier than hard/mixed targets. Exact-form head reconstruction residual checked above. Planned forward5: paired DECIDE all150 dev IDs, exact dev_decide_request.json. Current exact edit: scalar.weight=w_original+0.5*d_soft; d_soft saved candidate_weights/soft_direction.pt. All other tensors original. This is a directional readout change, not global scaling.")
```
