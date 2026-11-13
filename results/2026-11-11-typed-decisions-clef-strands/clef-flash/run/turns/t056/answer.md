Reactivating dormant neurons did not improve tuning accuracy, so I’m rejecting that change too. The strongest evidence still supports the original two-neuron correction plus the small targeted extension. I’ll now keep their design and hyperparameters fixed and recompute them using all available practice records. The former practice test set will then become fitting data—not an unbiased test—and hidden validation will judge the refit.


```python
# Hypothesis: more practice evidence can stabilize the already validated repair without changing its mechanism or chosen hyperparameters.
repair_remaining=splits["test"]
assert len(repair_remaining)==300
for num,part in [(1,repair_remaining[:200]),(2,repair_remaining[200:])]:
    req={"type":"read","model":"original","from":{"pass":1,"ids":part}}
    json.dump(req,open(WORK/f"read_refit_{num}.json","w"))
print("Refit READ sizes:",len(repair_remaining[:200]),len(repair_remaining[200:]))
print("Read file request shape:",type(json.load(open(WORK/"fwd_2"/"requests.json"))).__name__)
print("read_batch signature:",inspect.signature(read_batch))
print("Baseline normalized request example keys:",json.loads(base[0]["calls"][0]["prompt"]).keys())
with open(WORK/"notes.md","a") as f:
    f.write("\n## Dormant activation extension rejected\nRecentered 306 suppressed original neurons; joint linear+reactivated closed-form ridge reached at most 0.760 tune, below best_1 0.766 and best_2 0.786. No edits saved.\n")
    f.write("\n## Fixed-design full-practice refit plan\nRefit the already chosen hard-margin ridge 1e-4 shared direction at strength 0.5, and soft ridge .01 targeted field corrections, with no further hyperparameter search. Include all 900 practice items. The former 300 reserved test items are now fitting data; do not report subsequent practice scores as unbiased. Hidden validation remains unseen/aggregate only. Preserve best_1 and best_2 as fallbacks.\n")
    f.write("\n## Planned pass 12\nCurrent edits exactly best_2 (unchanged). READ original on first 200 former test IDs, read_refit_1.json (1000 questions), for fixed-design refit. No gold sent to model.\n")
```
