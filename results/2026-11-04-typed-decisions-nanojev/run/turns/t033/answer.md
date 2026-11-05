The controlled examples support partial—not perfect—cue recovery: changing violations from 0 to 1 raises repaired review probability from 45.8% to 73.6%, versus 83.4% to 83.7% originally. Some synthetic category examples still fail. I’ll test additional strengths and direction ablations on a fixed 40-case diagnostic panel, retaining the validated candidate regardless of these diagnostic results.

```python
# Hypothesis: removing or reversing the identified scalar direction should predictably undo its benefits.
panel_ids=fit_chunks[2][:40]
panel_order=[r["id"] for r in records if r["id"] in set(panel_ids)]
panel_requests=[json.loads(recbyid[i]["calls"][0]["prompt"]) for i in panel_order]
panel_requests += synthetic
jobs=[
    ("dose_zero",np.zeros(1024)),
    ("dose_quarter",.25*dw600),
    ("dose_half",.5*dw600),
    ("dose_threequarters",.75*dw600),
    ("dose_125",1.25*dw600),
    ("dose_150",1.5*dw600),
    ("cov_ge_1",Ua[:,ea>=1]@coef[ea>=1]),
    ("cov_ge_point1",Ua[:,ea>=.1]@coef[ea>=.1]),
    ("cov_ge_point01",Ua[:,ea>=.01]@coef[ea>=.01]),
    ("reverse_direction",-.5*dw600),
]
json.dump({"type":"read","model":"edited","requests":panel_requests},open(work/"panel_request.json","w"))
def install_job(j):
    label,dd=jobs[j]
    if np.linalg.norm(dd)==0: drop_edit("scalar.weight")
    else: save_edit("scalar.weight",torch.from_numpy((w0+dd).astype(np.float32))[None,:])
    with open(work/"notes.md","a") as f:
        f.write(f"\nPass {13+j}: diagnostic {label}; scalar original + specified dw600 ablation; 40 internal fitting cases plus 7 synthetic variants. Best snapshot best_1 untouched.\n")
    print("Installed",label)
install_job(0)
```
