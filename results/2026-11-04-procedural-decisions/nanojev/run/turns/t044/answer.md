The prior-only and high-variance ablations are already documented. Next I’ll reverse the full correction on the matched cue pairs: if it reads useful evidence, reversal should move cue-sensitive margins in the wrong direction. Then I’ll test half and 1.5× strength on the same probes.
```python
# Hypothesis: reversing the readout correction reverses sensitivity to decisive state changes.
save_edit("scalar.weight",torch.tensor(W0.reshape(-1)-delta900,dtype=torch.float32).reshape(1,-1))
with notes.open("a") as f:
    f.write("\nPass19: causal reversal scalar=original-delta900; matched 210 cue pairs, same probes as11/12, no fit. Pass20 planned half-strength and21 1.5x on same probes. Full validated edit backed up.\n")
probe_file=p/"cue_probe_requests.json"
json.dump({"type":"read","model":"edited","requests":probe_requests},probe_file.open("w"))
print(str(probe_file),len(probe_requests))
```
