The full-data targeted refit reached 76.53% validation accuracy, below the saved candidate’s 77.07%, so I’ll restore the saved candidate. I’ll use the remaining experiments to verify its mechanism—especially the three gated fields—rather than keep searching against validation scores.


```python
# Hypothesis: controlled state-cue probes can confirm that best_2's targeted gains come from evidence-sensitive readout, not generic option bias.
for name,arr in zip(["residual_scorer.0.weight","residual_scorer.0.bias","residual_scorer.3.weight"],repair_b2["tensors"]):
    save_edit(name,arr.reshape(SHAPES[name]))
for workflow in ["agent_trace_observability","customer_service"]:
    fit_ids=[ii for ii in splits["fit"] if byid[ii]["inputs"]["workflow"]==workflow]
    print("\nWorkflow:",workflow)
    for ii in fit_ids[:2]:
        print("ID",ii,"state",json.dumps(byid[ii]["inputs"]["state"],ensure_ascii=False))
    qrequest=json.loads(basebyid[fit_ids[0]]["calls"][0]["prompt"])
    print("Question schemas:",json.dumps({q:qrequest["questions"][q] for q in (["urgency"] if workflow.startswith("agent") else ["action","needs_human"])},ensure_ascii=False))
print("Probe metadata previous sample:",json.load(open(WORK/"causal_probe_meta.json"))[:3])
with open(WORK/"notes.md","a") as f:
    f.write("\n## Pass 15 complete; best_2 restored\nFixed-design full-data targeted refit: validation original 0.702000, edited 0.765333, gain +0.063333 ±0.009227 paired SE, n=300. Lower than saved best_2 (0.770667), so rejected as final. Restored best_2 exactly from targeted_candidate.pt. Remaining experiments focus on controlled cue response, extension dose/ablation, and exact reconstruction, not validation hyperparameter search.\n")
```
