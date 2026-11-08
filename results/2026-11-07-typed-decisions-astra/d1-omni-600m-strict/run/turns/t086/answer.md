Hidden validation confirms a gain of **2.27 percentage points ± 0.85 paired SE**. I’ll preserve this checkpoint, then check whether question-specific gating could remove the remaining payment/violation side effects. Any further edit must beat this checkpoint on independent practice data and pass validation.
```python
# Hypothesis: question gating may remove residual collateral effects; preserve the validated repair before testing that hypothesis.
import shutil, inspect, types
shutil.copytree(work / "edited", work / "best_2", dirs_exist_ok=True)
with open(work / "notes.md", "a") as f:
    f.write("\nPass21 hidden validation: original .5753333333, edited .598, "
            "paired gain +.0226666667 SE .0085033056;75 improved,43 worsened. "
            "Best validated checkpoint best_2 (identical to security_gated_moderate).\n"
            "Remaining passes may test localization of existing payment/violation directions; "
            "best_2 retained unless a new candidate passes independent confirmation and validation.\n")
print("HELPER SIGNATURES")
for key in ["edit_manifest","clear_edits","backup_edits","install_gated_security",
            "null_project_direction","protected_directions","agent_cue_direction"]:
    obj=globals().get(key)
    print(key, inspect.signature(obj) if callable(obj) else type(obj).__name__)
print("\nGATING HELPER")
try:
    print(inspect.getsource(install_gated_security))
except Exception as e:
    print(type(e).__name__)
print("\nFEATURE / SCORE VARIABLES")
for k,v in list(globals().items()):
    if not k.startswith("_") and (isinstance(v,np.ndarray) or isinstance(v,dict)) and any(s in k.lower() for s in ["feat","unit","row","acc","gold","meta","z2","z3","x2","x3","score","gate"]):
        print(k, v.shape if isinstance(v,np.ndarray) else ("dict",list(v)[:5]))
```
