The summary helper reads saved predictions from its metadata; I passed it the original metadata, which incorrectly showed zero change. I’ll rerun it with each pass’s matching metadata.
```python
# Hypothesis: matching each output with its own prediction metadata reveals the true question-level causal effects.
print("PAYMENT GAIN4 VS ORIGINAL")
read_summary(arr8, rowinfo8, arr3, rowinfo3)
print("\nPAYMENT4+VIOLATION2 VS PAYMENT4")
read_summary(arr10, rowinfo10, arr8, rowinfo8)
print("\nGATED SECURITY VS BASE")
read_summary(arr18, rowinfo18, arr10, rowinfo10)
```
