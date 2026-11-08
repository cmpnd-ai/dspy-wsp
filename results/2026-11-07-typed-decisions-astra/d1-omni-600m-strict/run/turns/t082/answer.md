The output was truncated before the most recent results. I’ll read just the end of the notes and list the relevant artifacts, so I don’t repeat a completed test.
```python
# Hypothesis: the final notes and pass-20 files identify the candidate awaiting validation.
print((work / "notes.md").read_text()[-4300:])
print("\nRECENT ARTIFACTS")
for p in sorted(work.iterdir(), key=lambda p:p.stat().st_mtime)[-18:]:
    print(p.name, p.stat().st_size)
print("\nRELEVANT VARIABLES")
print([k for k in globals() if not k.startswith("_") and any(s in k.lower() for s in ["edit","gate","confirm","score","split","sec","pair","meta","restor","checkpoint","request","dir","stat"])])
print("\nCURRENT EDITS", edited())
```
