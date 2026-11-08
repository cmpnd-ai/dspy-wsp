The smaller update also passes validation, but its gain is slightly below the first candidate’s. I’ll preserve both and inspect the original repair’s directions and remaining errors. Any further candidate will be selected using the development set, not validation labels.
```python
# Hypothesis: a few option-contrast modes may explain both the repair's gains and its remaining errors.
import shutil
with (work_dir / "notes.md").open("a") as f:
    f.write("\nPass 18 VALIDATION: actual-weight-ridge1 original .606667 -> edited .754000, paired gain +.147333 SE .014387; 165 improved, 41 worsened. Confirmed but below best_1 (.757333). Saved as best_weightnorm/. best_1 remains the highest validation score.\n")
if not (work_dir / "best_weightnorm").exists():
    shutil.copytree(work_dir / "edited", work_dir / "best_weightnorm")
print("Callable analysis helpers:", [(k, str(v)[:95]) for k,v in list(globals().items()) if callable(v) and not k.startswith("_") and k not in ["load","load_original","save_edit","drop_edit","tok"]])
print("Files:", sorted(q.name for q in work_dir.iterdir() if q.is_file()))
for name, obj in [("fit",fit), ("dev",dev), ("families", families), ("data",data)]:
    print(name, {k: (tuple(v.shape) if hasattr(v,"shape") else (type(v).__name__, len(v) if hasattr(v,"__len__") else None)) for k,v in obj.items()})
print("Best factor types:", type(A_best), type(A_selected), type(B))
print("Rank9 coefficients singular values:", np.linalg.svd(np.asarray(A_best), compute_uv=False))
print("read meta keys:", meta2.keys())
print("Example baseline keys:", baseline[0].keys())
print("Split lengths:", {k:len(v) for k,v in split.items()})
print("Example requests:", json.dumps(req2[0], ensure_ascii=False)[:5000])
```