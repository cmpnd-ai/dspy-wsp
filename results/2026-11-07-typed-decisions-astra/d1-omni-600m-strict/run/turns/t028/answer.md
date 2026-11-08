Gain 2 raised invoice-urgency accuracy from 38% to 52% on discovery half B. It changed no upstream features, and near-due invoices moved toward “elevated” much more than later-due invoices. The overall gain is still uncertain because of collateral changes. I’ll test gain 4 to establish the dose-response before deciding whether to keep or refine this direction.
```python
# Hypothesis: doubling the protected deadline readout will correct more overdue invoices, with an observable dose-response and measurable collateral cost.
note("Pass 7 payment gain2: discovery B accuracy .566 vs .557, paired +.0090 SE .00669 (26 improved,16 worse). Invoice urgency .38->.52; near-due elevated-routine mean -2.789->-.600, later -2.947->-2.462. Head features bit-identical. Invoice disposition -.06 and customer action +.08 reveal small collateral effects. Not yet sufficient for retention.")
payment_gain=4.0
install_linear_cues([(cue_neurons,d_payment,payment_gain)])
note("## Pass 8 planned: payment-direction dose response, gain 4\nExact same two-neuron direction as pass7, output coefficients now +/-4. "+json.dumps(edit_manifest())+
     "\nRead same 200 discovery-B cases/1000 questions via payment_read_b.json. Compare gains 0,2,4 for causal dose-response; confirmation remains unused.")
print(edit_manifest())
```
This pass tests gain 4 on the same discovery records, so I can distinguish stronger deadline correction from overcorrection and unrelated damage.
```forward
{"file": "/Users/dbreunig/Development/cmpnd/dspy-wsp/examples/runs/typed-decisions/d1-astra-strict/work/payment_read_b.json"}
```
