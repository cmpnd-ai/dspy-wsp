The frozen edit scores 77.87% across all practice records. I’ll calculate paired gains separately for fitting, development, and reserved-test cases, verify that the saved weights still match the chosen repair, and prepare the final validation check.

```python
# Hypothesis: the frozen repair's gains remain positive across workflows and the reserved practice split.
audit_original = {r["id"]: r for r in read_jsonl(WORK/"gen_1.jsonl")}
audit_edited = {r["id"]: r for r in read_jsonl(WORK/"gen_24.jsonl")}
assert set(audit_original) == set(audit_edited) == set(ex_by_id)

def paired_record_stats(ids_to_score):
    original_scores = np.array([audit_original[eid]["score"] for eid in ids_to_score], dtype=float)
    edited_scores = np.array([audit_edited[eid]["score"] for eid in ids_to_score], dtype=float)
    delta = edited_scores-original_scores
    return {
        "items": len(ids_to_score),
        "original": float(original_scores.mean()),
        "edited": float(edited_scores.mean()),
        "gain": float(delta.mean()),
        "paired_se": float(delta.std(ddof=1)/np.sqrt(len(delta))),
        "improved": int((delta>1e-10).sum()),
        "worsened": int((delta<-1e-10).sum())
    }

final_audit = {"all_practice": paired_record_stats(list(audit_original))}
for split_name, split_ids in splits.items():
    final_audit[split_name] = paired_record_stats(split_ids)
for workflow in sorted({e["inputs"]["workflow"] for e in examples}):
    wf_ids_audit = [e["id"] for e in examples if e["inputs"]["workflow"] == workflow]
    final_audit[workflow] = paired_record_stats(wf_ids_audit)
for name, stats in final_audit.items():
    print(name, json.dumps(stats))

# This audit must reproduce the previously measured fixed-candidate reserved-test result.
assert abs(final_audit["test"]["edited"] - np.mean(list(a13.values()))) < 1e-12
assert edited() == ["k.weight"]
assert torch.equal(load("k.weight"), best2_matrix)
assert hashlib.sha256((WORK/"edited/k.weight.npy").read_bytes()).hexdigest() == final_weight_sha256
with open(WORK/"final_practice_audit.json", "w") as f:
    json.dump(final_audit, f, indent=2)
with open(WORK/"notes.md", "a") as f:
    f.write("\nPass 24 frozen best2 final practice audit (no further tuning):\n")
    for name, stats in final_audit.items():
        f.write(f"- {name}: {json.dumps(stats)}\n")
    f.write("\n## Before pass 25\nExact final edit unchanged: best_2/edited/k.weight.npy, "
            "k.weight = K0 + outer(u,v_ref); log-gold floor .1, closed-form ridge .0003, "
            "strength .75, fit540 only. All other tensors original. "
            f"SHA256 {final_weight_sha256}. "
            "Final hidden VALIDATION both on all 300 cases. This repeats pass14 to verify the "
            "restored submission weights, not an independent statistical replication. "
            "No more fitting, selection, or weight changes are planned.\n")
```
