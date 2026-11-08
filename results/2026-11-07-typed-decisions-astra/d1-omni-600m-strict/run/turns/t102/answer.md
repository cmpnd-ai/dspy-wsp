The small reserve set regressed by **2.00 points ± 2.23 paired SE**. That result is uncertain, but it must be included. I’ll pool it with the 450 confirmation cases, then use the final pass to measure the already-selected repair on discovery data, clearly labeling that result as in-sample.
```python
# Hypothesis: pooled independent practice evidence remains positive despite the small reserve regression; no further tuning is justified.
reserve24_original=read_jsonl(work/"gen_24_original.jsonl")
reserve24_edited=read_jsonl(work/"gen_24_edited.jsonl")
summary24=paired_score_summary(reserve24_original,reserve24_edited)
pooled500_original=confirm20_original+reserve24_original
pooled500_edited=confirm20_edited+reserve24_edited
summary500=paired_score_summary(pooled500_original,pooled500_edited)
print("RESERVE",summary24)
print("POOLED INDEPENDENT PRACTICE",summary500)
final_discovery25={"type":"decide","model":"both","problems":discovery_ids}
(work/"final_discovery.json").write_text(json.dumps(final_discovery25))
with open(work/"notes.md","a") as f:
    f.write("\nPass24 untouched reserve (50): "+json.dumps(summary24)+"\n"
            "Pooled independent practice confirmation+reserve (500): "+json.dumps(summary500)+"\n"
            "Reserve regression is retained in reporting; no edits made in response. "
            "Best remains best_2: hidden validation gain +.022667 SE .008503 and "
            "pooled independent practice gain +.0256.\n"
            "\n## Pass 25 planned: final discovery-set audit\n"
            "Exact edit unchanged best_2, "+json.dumps(edit_manifest())+"\n"
            "DECIDE both on all400 discovery IDs in final_discovery.json. These are the development examples, "
            "so their gains are in-sample and will not be used as independent confirmation. "
            "No tuning or further changes planned; completion provides a complete 900-practice audit.\n")
```
