The closed-form repair predicts a development gain of 16.4 percentage points, with improvements concentrated in the diagnosed action, review, and severity errors. I’ll now test the prediction in the real model at half strength. The only candidate edit is a rank-nine addition to `layers.23.feed_forward.w2.weight`; both norm gains and the tied embedding remain original.
```python
# Hypothesis: a rank-nine W2 redirection at half strength produces the option-specific changes predicted by the recovered decision head.
A_selected=A_by_lambda[1.0].copy()
delta_w2=(torch.from_numpy(B).double() @
          (torch.from_numpy(A_selected).double() @ w2orig.double())).float()
print("Relative W2 update norm at unit strength:",float(delta_w2.norm()/w2orig.norm()))
print("Update rank bound:",rank)
print("A singular values:",np.linalg.svd(A_selected,compute_uv=False).round(4))
def install_w2(alpha,delta=None):
    if delta is None:delta=delta_w2
    for name in list(edited()):
        if name!=W2_NAME:drop_edit(name)
    save_edit(W2_NAME,w2orig+float(alpha)*delta)
def write_read_request(filename,ids,model="edited"):
    (WORK/filename).write_text(json.dumps({"type":"read","model":model,"from":{"pass":1,"ids":ids}}))
install_w2(0.5)
write_read_request("candidate_dev.json",split["dev"])
dose_predictions={}
for alpha in [0,0.5,1.0,1.5]:
    l,p=recompute_head(dev,A_selected,alpha)
    dose_predictions[alpha]=(l,p)
    print("Predicted dose",alpha,metrics_array(dev,p))
note(f"Proposed edit: W2' = W2 + alpha * B A W2, B is 2048x9 orthonormal option-contrast basis; A ridge=1 closed-form fit on 360 fit records. Rank <=9. Unit delta Frobenius / original = {float(delta_w2.norm()/w2orig.norm()):.6f}. All other tensors original. Targets depend on state/question features, not a global temperature.\n\nPlanned pass 11: READ edited dev=180 records, candidate_dev.json, alpha=0.5. Exact installed tensor W2_original + 0.5*delta_w2. Expected head-only dev metrics: "+str(metrics_array(dev,dose_predictions[0.5][1]))+". This begins actual-model dose-response verification.")
```
