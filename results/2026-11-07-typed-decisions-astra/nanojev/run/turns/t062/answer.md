The original cue-test read is complete. I’ll record the frozen edit and run the same requests with it, checking that the backbone features remain unchanged while the scalar head responds differently.
```python
# Hypothesis: only the scalar readout changes, so matched-request feature arrays should be identical before and after the repair.
assert edited() == ["scalar.weight"]
assert torch.equal(load("scalar.weight"), frozen_weight)
note("\nPlanned forward23: edited READ the exact same124 requests/268 questions as forward22, final_cue_edited_request.json. Exact edit frozen best4 scalar.weight=w0+.875*d_mix900, candidate_weights/mix900_0875_scalar.pt; all other tensors original. Verify identical head inputs and characterize directional cue responses, including failures. No weight fitting or selection.")
print("Frozen edit verified; matched requests:", len(diagnostic_requests), "questions:", num_diag_questions)
```
The frozen edit remains unchanged; this check will not select or fit new weights.