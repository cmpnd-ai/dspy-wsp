The discovery set shows several systematic errors: invoice urgency defaults to “routine,” agent risk defaults to “benign,” and security severity defaults to level 2. I’ll read activations for 200 discovery records to test whether the head contains useful option distinctions that its scoring path underweights.
```python
# Hypothesis: the shared option-scoring path underweights evidence-supported alternatives to benign/routine defaults.
read_ids_a = []
read_ids_b = []
for wf in sorted({e["inputs"]["workflow"] for e in examples}):
    ids = [eid for eid in discovery_ids if by_id[eid]["inputs"]["workflow"] == wf]
    read_ids_a += ids[:50]
    read_ids_b += ids[50:]
for key in [("invoice_processing","urgency"), ("customer_service","action")]:
    errors = sorted((r for r in groups[key] if not r["ok"]), key=lambda r:-max(r["p"].values()))
    for rr in errors[:2]:
        print("\n", rr["id"], key)
        print(json.dumps(by_id[rr["id"]]["inputs"]["state"],ensure_ascii=False)[:1900])
        print("P",rr["p"],"G",rr["g"])
print("\nScorer weight statistics")
for n in ["head.scorer.0.weight","head.scorer.1.weight","head.scorer.1.bias","head.scorer.3.weight"]:
    t = load_original(n)
    print(n, "norm",float(t.norm()), "mean",float(t.mean()), "std",float(t.std()), "range",(float(t.min()),float(t.max())))
read_req = {"type":"read","model":"original","from":{"pass":1,"ids":read_ids_a},"layers":[14,16]}
(WORK / "read_a.json").write_text(json.dumps(read_req))
note("## Pass 2 planned: discovery activations A\nExact edits: none. Read original on 200 discovery cases (50/workflow, first half of discovery; read_a.json), all 1000 questions, layers 14 and 16. Hypothesis: systematic benign/routine/level-2 defaults reflect a shared option-scoring readout mismatch. Discovery accuracies: security severity .33; agent risk .40; invoice discrepancy .40; invoice urgency .41; customer action .44. Need inspect returned activations before selecting a component.")
print("\nRead request:",WORK / "read_a.json")
```
