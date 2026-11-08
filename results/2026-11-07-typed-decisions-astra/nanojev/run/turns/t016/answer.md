The actual half-strength edit matches the prediction: development accuracy rises from 37.3% to 52.0%, a paired gain of 14.7 ± 1.8 percentage points. I’ll now test full strength on the same records to establish the predicted dose-response.
```python
# Hypothesis: the same readout direction at full strength will outperform the half-strength intervention.
note("Forward5 actual half-strength soft direction on dev150: original0.373333, edited0.520000, paired gain0.146667 SE0.018114, 82 improved/19 worsened. Matches cached head prediction exactly; causal evidence for scalar direction.")
w_full=set_scalar_direction(d_soft,1.0)
torch.save(w_full,candidate_dir/"soft_full_scalar.pt")
note("Planned forward6: paired DECIDE same150 dev IDs, dev_decide_request.json. Exact current edit: scalar.weight=w_original+1.0*d_soft (candidate_weights/soft_full_scalar.pt). All other tensors original. Prediction: dev~0.5853, larger gain than strength0.5.")
print("Current edits",edited(),"direction norm",float(np.linalg.norm(d_soft)))
```
