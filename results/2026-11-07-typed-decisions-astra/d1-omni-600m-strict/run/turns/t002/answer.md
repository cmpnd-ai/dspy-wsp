I’ll run the original model on all 900 practice records. I’ll use a fixed 400-record discovery split for diagnosis and reserve 450 records for paired confirmation; those reserved records won’t be used to derive edits.
```python
# Hypothesis: a fixed discovery/confirmation split will separate mechanism discovery from evidence of generalization.
rng = random.Random(7349)
discovery_ids, confirmation_ids, reserve_ids = [], [], []
for wf in sorted({e["inputs"]["workflow"] for e in examples}):
    ids = [str(e["id"]) for e in examples if e["inputs"]["workflow"] == wf]
    rng.shuffle(ids)
    discovery_ids += ids[:100]
    confirmation_ids += ids[100:212]
    reserve_ids += ids[212:]
confirmation_ids += reserve_ids[:2]
reserve_ids = reserve_ids[2:]
split = {"discovery": discovery_ids, "confirmation": confirmation_ids, "reserve": reserve_ids}
(WORK / "splits.json").write_text(json.dumps(split))
base_req = {"type": "decide", "model": "original", "problems": [str(e["id"]) for e in examples]}
(WORK / "baseline_request.json").write_text(json.dumps(base_req))
note("## Pass 1 planned: baseline\nExact edits: none. Original model on all 900 practice IDs in baseline_request.json. Fixed split (splits.json): 400 discovery, 450 confirmation, 50 reserve. Inspect individual failures only on discovery; confirmation records not used to derive directions.")
print("Split sizes:", {k: len(v) for k, v in split.items()})
print("Baseline request:", WORK / "baseline_request.json")
```
