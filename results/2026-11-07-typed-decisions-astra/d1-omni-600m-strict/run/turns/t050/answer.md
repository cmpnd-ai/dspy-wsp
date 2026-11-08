The repair gained **2.04 percentage points ± 0.72 points SE** on the independent practice cases. I’ll now check it on the hidden validation set before treating it as a fallback checkpoint.
```python
# Hypothesis: the independently confirmed payment/violation repair should also improve unseen validation cases.
note("\nPass12 independent confirmation (450 cases): original .564889, edited .585333, paired gain +.020444 SE .007220; 103 improved,64 worsened. No confirmation records inspected to construct edits.")
backup_edits("confirmed_practice_payment4_violation2")
note("\n## Pass 13 planned: hidden validation of initial repair\nExact edits unchanged: payment4_violation2, "+json.dumps(edit_manifest())+
     "\nDECIDE both on all 300 hidden validation cases; aggregate confirmation only. No validation examples accessible.")
print("Edits recorded; validation next.")
print("Discovery-only security errors:")
for qid in sorted({r["q"] for r in disc_rows if r["wf"]=="security_incidents"}):
    rows_q = [r for r in disc_rows if r["wf"]=="security_incidents" and r["q"]==qid]
    wrong = [r for r in rows_q if not r["ok"]]
    print(qid, "accuracy",round(np.mean([r["ok"] for r in rows_q]),3),
          "errors",collections.Counter((r["gold"],r["pred"]) for r in wrong).most_common(6))
```
