I’ll quantify the confusions and reserve disjoint practice subsets for fitting, choosing edit strength, and final testing. I’ll also read several failed records to distinguish missing state cues from simple confidence errors.

```python
# Hypothesis: the pointer head systematically misreads some state-dependent distinctions despite recognizing the workflow.
recbyid = {str(r["id"]): r for r in records}
rows = []
for r in records:
    ex = byid[str(r["id"])]
    ans = json.loads(r["calls"][0]["completion"])
    for qid, gold in ex["labels"]["gold"].items():
        p = ans[qid]["probabilities"]
        target = max(gold["probabilities"], key=gold["probabilities"].get)
        best = max(p, key=p.get)
        rows.append({"id": str(ex["id"]), "wf": ex["inputs"]["workflow"], "qid": qid,
                     "type": ex["labels"]["questions"][qid]["type"], "p": p,
                     "gold": gold["probabilities"], "best": best, "target": target,
                     "ok": best == target, "confidence": p[best]})
groups = collections.defaultdict(list)
for r in rows:
    groups[(r["wf"], r["qid"])].append(r)
for (wf, qid), rr in sorted(groups.items()):
    conf = collections.Counter((r["target"], r["best"]) for r in rr if not r["ok"])
    print(f"{wf:26s} {qid:22s} acc={np.mean([r['ok'] for r in rr]):.3f} "
          f"pred_conf={np.mean([r['confidence'] for r in rr]):.3f} "
          f"gold_conf={np.mean([max(r['gold'].values()) for r in rr]):.3f}")
    print("  wrong gold -> pred:", conf.most_common(4))
rng = np.random.default_rng(2610)
splits = {"fit": [], "dev": [], "test": []}
for wf in sorted({ex["inputs"]["workflow"] for ex in examples}):
    ids = np.array([str(ex["id"]) for ex in examples if ex["inputs"]["workflow"] == wf])
    rng.shuffle(ids)
    for name, part in zip(splits, [ids[:135], ids[135:180], ids[180:]]):
        splits[name].extend(part.tolist())
json.dump(splits, open(WORK / "splits.json", "w"))
id_split = {i: s for s, ids in splits.items() for i in ids}
print("\nFixed split:", {k: len(v) for k,v in splits.items()})
for wf, qid in [("agent_trace_observability","outcome"),("agent_trace_observability","risk"),
                 ("invoice_processing","matches_order"),("customer_service","urgency"),
                 ("security_incidents","severity")]:
    wrong = [r for r in groups[(wf,qid)] if not r["ok"] and id_split[r["id"]] == "fit"]
    wrong.sort(key=lambda r: r["confidence"], reverse=True)
    for r in wrong[:2]:
        print("\nFAIL", r["id"], wf, qid, "pred", r["p"], "gold", r["gold"])
        print(json.dumps(byid[r["id"]]["inputs"]["state"], ensure_ascii=False)[:1700])
lognote("Fixed stratified practice split (seed 2610): fit=540 (135/workflow), dev=180, test=180. "
        "IDs in splits.json. Test labels/outputs will not be used for fitting or strength selection. "
        "All baseline aggregate groups inspected, but only fit individual failures inspected.")
lognote("Baseline per question:\n" + "\n".join(
    f"- {wf}/{qid}: {np.mean([r['ok'] for r in rr]):.4f}"
    for (wf,qid),rr in sorted(groups.items())))
```
