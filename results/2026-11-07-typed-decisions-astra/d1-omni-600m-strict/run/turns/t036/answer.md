The refined violation direction separates violated from non-violated traces on discovery half B, while its mean response to the clean and tool-error controls is near zero. I’ll add it at gain 2 alongside the payment repair, using another pair of nearly unused scorer neurons.
```python
# Hypothesis: strengthening a violation-specific readout will correct harmful/high/critical judgments only when constraint violations are present.
violation_neurons=(518,33)
violation_gain=2.0
install_linear_cues([(cue_neurons,d_payment,4.0),(violation_neurons,d_violation2,violation_gain)])
note("## Pass 10 planned: add violation readout, gain 2\nExact edits: payment cue gain4 in neurons594/746 plus refined violation cue gain2 in neurons518/33. Refined direction uses violation1/3 minus irreversible-only matched traces; nulls clean/dry/tool-error/irreversible target means, all other agent question means across variants, other workflow means and 32 unrelated contrast PCs. Direction residual norm .0443; discovery-B harmful-partial projection .0299 for violations versus .0026 without. "+json.dumps(edit_manifest())+
     "\nRead 200 discovery-B cases/1000 questions via payment_read_b.json. Predict selective improvement on agent outcome/risk/urgency. Compare both original pass3 and payment-only pass8.")
print(edit_manifest())
```
