The weight matrix itself is not rank-one; the compression appears in its option-varying activations. I’ll avoid overstating that diagnosis. I’ll now compare corrections for the specific failure fields and read the edited tuning activations to verify that the saved weights behave exactly as the local simulation predicts.

```python
# Hypothesis: the validated bypass changes only final option margins; a few remaining fields may benefit from targeted rather than global corrections.
print("Targeted field diagnostics on tune (baseline, shared bypass, field-soft, field-hard, interaction-soft):")
diag_names = [("agent_trace_observability","outcome"),("agent_trace_observability","urgency"),
              ("security_incidents","credential_compromise"),("invoice_processing","duplicate"),
              ("invoice_processing","disposition"),("invoice_processing","urgency")]
Ls = [
    D["logits"],
    edited_pair_logits(.5),
    D["logits"]+conditional_candidates["soft",.01]["correction"],
    D["logits"]+.5*conditional_candidates["hard",.01]["correction"],
    D["logits"]+qr_candidates["soft",.001]["correction"]
]
for qkey in diag_names:
    ix=torch.tensor([k==qkey for k in D["qkey"]])&D["istune"]
    print(qkey,[round(summary_logits(L,ix)["accuracy"],3) for L in Ls])
# Check less regularized output-only reads before attributing the issue to nonlinear compression.
print("Weakly regularized existing-activation correction (soft target, strength .5):")
for ratio in [1e-8,1e-7,1e-6,1e-5]:
    d=(evecs@((evecs.T@rhs)/(evals+ratio*ridge_unit))).float()
    w=(wout+.5*d).to(torch.bfloat16).float()
    L=D["logits"]+D["C"]@(w-wout)
    print(ratio,"norm",round(d.norm().item(),1),"fit/tune",round(summary_logits(L,D["isfit"])["accuracy"],4),summary_logits(L,D["istune"]))
json.dump({"type":"read","model":"edited","from":{"pass":1,"ids":splits["tune"]}},
          (WORK/"read_edited_tune.json").open("w"))
note("Weight SVD does not support a rank-one matrix claim (top singular value accounts for only 9.1% of W0 energy); compression diagnosis refers specifically to option-centered *activations*. 335/1024 baseline neurons contribute less than 1e-5 RMS to option logits.\n## Planned pass 7\nExact edit unchanged from best_1: rows 791,1015 half-strength linear bypass. READ edited on 100 tune records, read_edited_tune.json, to verify exact logit response and that upstream features are unchanged.")
```
