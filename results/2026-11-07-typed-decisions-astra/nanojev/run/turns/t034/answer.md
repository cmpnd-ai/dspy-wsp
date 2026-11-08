The refitted readout validates at 57.4%, a gain of 22.6 ± 1.5 percentage points over the original. I’ll preserve it as the new best checkpoint and investigate whether the choice-only set head can recover additional distinctions that the scalar projection still misses.
```python
# Hypothesis: residual choice errors may be recoverable from the set head's nonlinear features without changing the backbone.
note("Forward13 mix600_075 hidden validation300: original0.348000, edited0.574000, gain0.226000 SE0.014991;207 improved/32 worsened. New best confirmed candidate; its +1.4pp vs best1 on validation is not claimed statistically significant because only aggregate results are available.")
best2=WORK/"best_2"
if best2.exists(): shutil.rmtree(best2)
shutil.copytree(WORK/"edited",best2)
torch.save({"scalar.weight":load("scalar.weight").clone()},WORK/"best_2_weights.pt")
# Examine additional scalar-input conventions in the permitted set head.
for sname,ss in [
    ("ones",torch.ones_like(base_logits)),("logcount",cmask.sum(1).float().log()[:,None,None].expand_as(base_logits)),
    ("count",cmask.sum(1).float()[:,None,None].expand_as(base_logits)),
    ("mean_logit",(base_logits.squeeze(-1)*cmask).sum(1)[:,None,None]/cmask.sum(1)[:,None,None]),
    ("scaled_raw",base_logits/4),
]:
    if ss.shape!=base_logits.shape: ss=ss.expand_as(base_logits)
    pred=set_formula(H,ss,activations["identity"],4,"project",activations["tanh"])
    diff=(pred-target_delta)[cmask]
    print("scalar feature",sname,"RMSE",float(diff.square().mean().sqrt()),"max",float(diff.abs().max()))
print("set_project last column",head0["set_project.weight"][:,-1].norm().item(),
      "other col avg norm",head0["set_project.weight"][:,:-1].norm(dim=0).mean().item())
print("Set output residual candidates will use original-vs-edited READ to verify formulas.")
# Read only choices from all600 available hidden-feature records; total900 questions fits one pass.
choice600_ids=splits["fit"]+splits["dev"]
choice600_requests=[];choice600_manifest=[]
for pid in choice600_ids:
    req=json.loads(baseline_by_id[pid]["calls"][0]["prompt"])
    req["questions"]={q:v for q,v in req["questions"].items() if v["type"]=="choice"}
    choice600_requests.append(req);choice600_manifest.append(pid)
(WORK/"choice600_manifest.json").write_text(json.dumps(choice600_manifest))
(WORK/"choice600_read_request.json").write_text(json.dumps({"type":"read","model":"edited","requests":choice600_requests}))
note("Planned forward14: edited READ only choice questions for fit450+dev150, exact choice600_read_request.json and choice600_manifest.json, 900 questions. Exact edit scalar.weight=w_original+.75*d_mix600 (best2), all other tensors original. Purpose: measure actual set correction under validated edit before any set-head intervention.")
print("Choice questions",sum(len(r["questions"]) for r in choice600_requests))
```
