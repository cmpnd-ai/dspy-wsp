At 1.5× strength, accuracy is 78.07% on this cohort. I’ll test half strength next, then restore the validated full-strength edit for the final comparison and validation.
```python
# Hypothesis: a partial correction leaves useful evidence underweighted.
save_edit("scalar.weight",torch.tensor(W0.reshape(-1)+.5*delta900,dtype=torch.float32).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass22 1.5x on450: .47185→.78074 gain .30889±.02394. Pass23 half-strength scalar original+.5delta900, same450; diagnostic only.\n")
```
