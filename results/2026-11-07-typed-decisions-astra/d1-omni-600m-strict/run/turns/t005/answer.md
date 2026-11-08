I’ll count predicted-versus-gold confusions for each discovery question, including confidence. Then I’ll read representative failures from the weakest questions to look for a common cause.
```python
# Hypothesis: a few stable option confusions account for much of the baseline's 43% error rate.
base_by_id = {str(r["id"]): r for r in baseline_records}
def answers(record):
    out = {}
    for call in record["calls"]:
        completion = json.loads(call["completion"])
        for q, ans in completion.items():
            if "probabilities" in ans:
                out[q] = {str(k): float(v) for k, v in ans["probabilities"].items()}
            elif "noul" in ans:
                out[q] = {"false": 1-float(ans["noul"]), "true": float(ans["noul"])}
    return out
def top(probs):
    return max(probs, key=probs.get)
disc_rows = []
for eid in discovery_ids:
    r, ex = base_by_id[eid], by_id[eid]
    for qid, p in answers(r).items():
        g = ex["labels"]["gold"][qid]["probabilities"]
        disc_rows.append({"id":eid, "wf":ex["inputs"]["workflow"], "q":qid, "p":p, "g":g,
                          "pred":top(p), "gold":top(g), "ok":top(p)==top(g)})
groups = collections.defaultdict(list)
for r in disc_rows:
    groups[(r["wf"],r["q"])].append(r)
for key, rows in sorted(groups.items(), key=lambda kv: np.mean([r["ok"] for r in kv[1]])):
    errors = [r for r in rows if not r["ok"]]
    print(key, "acc",round(np.mean([r["ok"] for r in rows]),3),
          "confusion",collections.Counter((r["gold"],r["pred"]) for r in errors).most_common(5),
          "error_conf",round(np.mean([max(r["p"].values()) for r in errors]),2) if errors else None)
print("\nDiscovery average:", np.mean([r["ok"] for r in disc_rows]))
weak_keys = [k for k,rs in sorted(groups.items(), key=lambda kv:np.mean([r["ok"] for r in kv[1]]))[:5]]
for key in weak_keys:
    errors = [r for r in groups[key] if not r["ok"]]
    errors.sort(key=lambda r: -max(r["p"].values()))
    print("\nRepresentative errors", key)
    for r in errors[:2]:
        print(r["id"],json.dumps(by_id[r["id"]]["inputs"]["state"],ensure_ascii=False)[:1700])
        print("P", r["p"], "G",r["g"])
note("Baseline pass 1: accuracy 0.56556, item SE 0.00714 on 900. Discovery question breakdown computed; no edits. Confirmation records not inspected.")
```
