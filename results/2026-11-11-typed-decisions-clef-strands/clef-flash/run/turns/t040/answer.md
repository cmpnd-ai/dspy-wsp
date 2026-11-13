Analytically correcting bfloat16 offsets raises the targeted extension to 78.6% tuning accuracy. I’ll test that concrete edit on the reserved practice records. It adds 22 gated neurons for just three fields; the validated two-neuron edit remains backed up.

```python
# Hypothesis: correcting numerical offsets allows narrowly gated state-evidence reads to improve three fields without broad interference.
targeted_candidate=rounding_trials[32.]
def save_scorer_tensors(tensors):
    wf,bf,wl=tensors
    save_edit("residual_scorer.0.weight",wf)
    save_edit("residual_scorer.0.bias",bf)
    save_edit("residual_scorer.3.weight",wl.reshape(1,-1))
save_scorer_tensors(targeted_candidate)
torch.save({"tensors":targeted_candidate,"neurons":all_modified,"qkeys":target_qkeys},
           WORK/"targeted_candidate.pt")
Ltarget=simulate_scoring_tensors(targeted_candidate)
change=center_options(Ltarget-edited_pair_logits(.5),D["mask"])
print("Tune target/non-target correction RMS:",
      change[targetmask&D["istune"]].square().mean().sqrt().item(),
      change[(~targetmask)&D["istune"]].square().mean().sqrt().item())
print("Tune performance:",summary_logits(Ltarget,D["istune"]))
json.dump({"type":"decide","model":"both","problems":splits["test"]},
          (WORK/"test_targeted_extension.json").open("w"))
note("## Planned pass 10\nExact edit: best_1 two-neuron bypass plus 22 gated neurons "+str(new_neurons)+
     " for agent urgency, customer action/needs_human. Gate strength 32, expert strength 1, bfloat16 offset compensation computed once from fit activations (no gold used in compensation). Tensors saved targeted_candidate.pt. Tune accuracy 0.786 vs best_1 0.766. DECIDE both on fixed reserved 300 practice test IDs. No test labels used to fit or select this candidate. best_1 remains backed up.")
```

This tests whether the targeted extension generalizes; it does not replace the validated fallback unless the held-out evidence supports it.