The expanded fit reaches 75.80% on validation: +15.13 points over the original, with a paired SE of 1.45 points. I’ll preserve it as the current best. With the remaining budget, I’ll test one final, fixed-recipe expansion to 720 practice records, leaving 180 out of coefficient fitting.
```python
# Hypothesis: additional independently sampled write vectors improve the fixed rank-nine repair without changing its target or strength.
with (work_dir/"notes.md").open("a") as f:
    f.write("\nPass 20 VALIDATION CONFIRMED expanded540: original .606667 -> edited .758000; paired gain +.151333 SE .014527; 170 improved, 42 worsened. Highest validation score so far, though difference from best_1 is negligible. Saved best_expanded540/.\n")
    f.write("\nFinal experiment predeclared: add 180 randomly selected, workflow-balanced records from the existing 360 practice-test records to the closed-form fit. Keep basis, log-probability target, ridge1 and alpha1 fixed, with no strength grid. Three READ passes recover their final-write vectors; one DECIDE assesses the other 180 records; one final hidden-validation pass. Coefficients then use 720 practice records only. Earlier 540 candidate remains safe.\n")
if not (work_dir/"best_expanded540").exists():
    shutil.copytree(work_dir/"edited", work_dir/"best_expanded540")
rng_final = np.random.default_rng(71032)
expand_ids, final_check_ids = [], []
for wf in sorted({by_id[i]["inputs"]["workflow"] for i in split["test"]}):
    wf_test = [i for i in split["test"] if by_id[i]["inputs"]["workflow"] == wf]
    wf_test = list(rng_final.permutation(wf_test))
    expand_ids.extend(wf_test[:45])
    final_check_ids.extend(wf_test[45:])
json.dump({"added_fit":expand_ids,"remaining_check":final_check_ids},
          (work_dir/"final_expansion_split.json").open("w"), indent=2)
print("New split:",len(expand_ids),len(final_check_ids))
print("call_probs output sample:", str(call_probs(records19[0]))[:2200])
for func in [read_arrays, infer_subset, combine_data, temp_for_data]:
    print(func.__name__, inspect.signature(func), "variables",func.__code__.co_varnames,
          "globals",func.__code__.co_names)
print("Existing read request:",json.loads((work_dir/"read_fit_a.json").read_text()).keys())
print("Arrays metadata:",json.dumps(meta2["files"],ensure_ascii=False)[:2200])
```