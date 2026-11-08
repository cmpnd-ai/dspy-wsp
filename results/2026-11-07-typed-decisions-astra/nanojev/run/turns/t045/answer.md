I’ll append the final 300 records using the existing loader, which checks cached probabilities against the original program outputs. These records will now be used for fitting, so they cannot serve as an independent test for the new candidate.
```python
# Hypothesis: a larger sample will stabilize the same under-read evidence direction without requiring another edited component.
print("Existing read structure:", {k:(type(v).__name__,len(v)) for k,v in reads.items()})
print("A arrays:", {k:v.shape for k,v in A.items()})
print("Gold schema:", {k:str(v)[:500] for k,v in examples[0]["labels"].items()})
print("Array sample:", "G",G[0].round(3), "mask",mask[0], "RW",RW[0],
      "Tmix",Tmix[0].round(3), "L0",L0[0].round(3))
new15 = load_read_dataset(15, read15_ids)
new16 = load_read_dataset(16, read16_ids)
print("New loader result:", type(new15).__name__,
      [(type(v).__name__,len(v)) for v in new15] if isinstance(new15,tuple) else list(new15))
print("Current best relative change:", float(torch.linalg.vector_norm(load("scalar.weight")-load_original("scalar.weight")) /
                                       torch.linalg.vector_norm(load_original("scalar.weight"))))
note("\nForward16 completed final100 original READs. All900 practice features are now available. The earlier test300 is retired for new fitting; frozen-candidate forward7/12 estimates remain independent. Next: fixed-recipe full900 solve, not hyperparameter selection on test labels.")
```
