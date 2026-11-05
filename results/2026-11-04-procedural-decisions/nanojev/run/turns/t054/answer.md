Half strength reaches 63.70%, substantially below the stronger edit. I’ll restore full strength and measure it on the identical cohort before the final validation check.
```python
# Hypothesis: the validated full correction balances evidence recovery better than half or overshoot strength.
save_edit("scalar.weight",torch.tensor(W0.reshape(-1)+delta900,dtype=torch.float32).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass23 half on450 .47185→.63704 gain .16519±.01458. Pass24 restored full scalar original+delta900, same450 paired comparison. All other tensors original.\n")
print("Restored full; difference from saved edit:",
      np.max(np.abs(load("scalar.weight").numpy().reshape(-1)-(W0.reshape(-1)+delta900))))
```
