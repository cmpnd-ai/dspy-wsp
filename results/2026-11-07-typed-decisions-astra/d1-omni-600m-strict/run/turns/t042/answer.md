I’ll recover the existing analysis helpers and check that the read-pass rows match the correct practice cases. This avoids attributing a change to the wrong example or question.
The helper signatures will also show how to summarize paired changes without reimplementing any model computation.
```python
# Hypothesis: the existing paired-read summaries can separate useful violation repairs from off-target ranking changes.
import inspect
for fname in ["load_read", "read_summary", "install_linear_cues", "edit_manifest", "backup_edits", "note", "unit_center_features", "deadline_summary"]:
    fn = globals()[fname]
    print(fname, inspect.signature(fn))
    try:
        src = inspect.getsource(fn)
        print(src[:4200])
    except (OSError, TypeError):
        pass
print("rowinfo3 sample:", rowinfo3[:2])
print("meta3 type/keys:", type(meta3), list(meta3)[:8])
print("array keys:", list(arr3))
print("read_ids_b:", len(read_ids_b), read_ids_b[:3])
print("example:", json.dumps(by_id[read_ids_b[0]], ensure_ascii=False)[:4200])
```
