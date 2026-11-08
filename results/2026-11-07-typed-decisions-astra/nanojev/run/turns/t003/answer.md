The original model scores 36.4%. I’ll break that down by workflow and question, inspect representative failures, and check whether the scalar head or choice-only set-attention correction could explain them.
```python
# Hypothesis: option priors or a misaligned scalar readout explain systematic, high-confidence errors.
baseline = [json.loads(s) for s in (WORK/"gen_1.jsonl").read_text().splitlines()]
print("Record keys:", list(baseline[0]))
print("First scored record:", json.dumps(baseline[0], ensure_ascii=False)[:9700])
print("Head weight summaries:")
for n,t in head0.items():
    print(n, "mean", round(t.mean().item(),5), "std", round(t.std().item(),5) if t.numel()>1 else 0,
          "norm", round(t.norm().item(),4), "max_abs", round(t.abs().max().item(),4))
note("Forward 1 original all 900: score 0.364222, item SE 0.007638. 890/900 items imperfect. Next: inspect per-question failures and decompose head logits.")
```
