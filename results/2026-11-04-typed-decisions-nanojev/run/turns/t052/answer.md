Next I’ll test 1.5× strength. A plateau or decline would distinguish a useful readout correction from an unbounded confidence increase.

```python
# Hypothesis: excessive correction eventually reduces panel accuracy.
install_job(5)
with (work/"notes.md").open("a") as f:
    f.write("\nPass 19: diagnostic scalar original + 1.5*dw600, same fixed fitting panel and synthetic cues; best_1 unchanged.\n")
```