The cached read outputs are available. I’ll compare the payment and violation interventions separately, using only discovery cases, to decide which questions should remain exposed to each cue.
```python
# Hypothesis: separate causal interventions identify which readouts warrant gating and which already generalize usefully.
print("INFO TABLES", [(k,len(v)) for k,v in list(globals().items())
                      if isinstance(v,list) and len(v)==1000])
print("INFO SAMPLE", discovery22_base[1][0])
print("\nPAYMENT GAIN4 VS ORIGINAL")
read_summary(arr8, discovery22_base[1], arr3, discovery22_base[1])
print("\nPAYMENT4+VIOLATION2 VS PAYMENT4")
read_summary(arr10, discovery22_base[1], arr8, discovery22_base[1])
```
