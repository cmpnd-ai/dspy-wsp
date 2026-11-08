The low-dose edit raised security severity accuracy from 38% to 50% on discovery B. It also moved the high-versus-medium margin upward for production-access accounts, but not for other accounts. I’ll double the security readout strengths to test the dose response and check for overshoot.
```python
# Hypothesis: stronger security cue readouts should repair remaining high/critical defaults, but may expose collateral errors.
note("\nPass15 security low (crown2/prod2/high1): discovery B accuracy .596, gain vs original +.0390 SE .01019; increment vs base repair +.0080 SE .00692. Security severity .38->.50. Production-access 3-2 margin -.790->+.092; unprivileged -.634->-.679. Some customer/action collateral (-.06) and invoice/disposition improvement (+.06). Checkpoint security_low.")
sec_cues_high=[(security_neuron_pairs[0],d_crown,4.0),
               (security_neuron_pairs[1],d_prod,4.0),
               (security_neuron_pairs[2],d_high,2.0)]
install_linear_cues(base_cues+sec_cues_high)
note("\n## Pass 16 planned: security readout dose response, high\nExact edit: payment4+violation2 unchanged, security crown4/prod4/highasset2, same six rows/directions as pass15. "+json.dumps(edit_manifest())+
     "\nRead 200 discovery-B cases/1000 questions via payment_read_b.json. Compare security strengths0/low/high; no confirmation data used to select strength.")
print(edit_manifest())
```
