The violation readout behaves as predicted: on the 10 violated traces in discovery half B, harmful-outcome accuracy rose from 0/10 to 9/10, and high-risk accuracy rose from 6/10 to 10/10. The combined edit gained 3.1 percentage points overall (paired SE 1.01 points). I’ll test violation gain 4 to check whether further strength helps or starts damaging non-violated cases.
I’ll keep the payment gain fixed at 4 and double only the violation readout. This isolates the dose-response of the violation component.
```python
# Hypothesis: a stronger violation readout may resolve the remaining violated trace, but should reveal overcorrection if gain 2 is already sufficient.
note("Pass 10 payment4+violation2: discovery B .588 vs .557, paired +.0310 SE .01013 (47 improved,23 worse). Violated traces n10: outcome 0/10 original ->9/10 edited; risk 6/10->10/10; urgency4/10->7/10. Nonviolated outcome26/40->24/40, risk14/40->16/40. Checkpoint payment4_violation2.")
violation_gain=4.0
install_linear_cues([(cue_neurons,d_payment,4.0),(violation_neurons,d_violation2,violation_gain)])
note("## Pass 11 planned: violation dose response, gain4\nPayment cue remains gain4; violation cue gains now +/-4 (same directions and rows). "+json.dumps(edit_manifest())+
     "\nRead same 200 discovery-B records/1000 questions, payment_read_b.json. Compare violation gains0/2/4; look for saturation and collateral effects.")
print(edit_manifest())
```
