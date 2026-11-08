The three strengths provide a direct causal test. I’ll record the 1.5× result, restore unit strength, and run the full program on the 360 reserved practice-test records, which were not used to fit the coefficients or select their strength.
```python
# Hypothesis: the unit-strength rank-nine repair yields a statistically clear gain on practice records excluded from fitting and selection.
r13=read_arrays(13)
print("Actual 1.5-strength metrics:",metrics_array(dev,r13["p"]))
print("Dose response:",[(a,metrics_array(dev,r["p"])["acc"]) for a,r in [(0,dev),(0.5,r11),(1,r12),(1.5,r13)]])
install_w2(1.0)
import shutil
backup=WORK/"candidate_rank9_ridge1"
if backup.exists():shutil.rmtree(backup)
shutil.copytree(WORK/"edited",backup)
np.savez(WORK/"candidate_rank9_ridge1_factors.npz",B=B,A=A_selected,alpha=1.0,ridge=1.0)
(WORK/"test_pair.json").write_text(json.dumps({"type":"decide","model":"both","problems":split["test"]}))
note("Pass 13 actual 1.5-strength dev: "+str(metrics_array(dev,r13["p"]))+
     ". Actual dose-response accuracies: original .618889, alpha=.5 .740000, alpha=1 .783333, alpha=1.5 .710000. Causal optimum is bounded; excessive correction reverses gains.\n\nCandidate snapshot: candidate_rank9_ridge1/ (not yet validation-confirmed).\n\nPlanned pass 14: DECIDE both on all 360 reserved practice-test IDs, test_pair.json. Exact current edit: W2_original + B A_ridge1 W2_original, alpha=1, rank<=9; all other tensors original. Test records were excluded from fit and strength/regularization selection.")
print("Next:",str(WORK/"test_pair.json"),"backup:",str(backup))
```
