The metadata listing was dominated by row names. I’ll print only the array descriptions and file names so I can choose an edit supported by the data actually returned.

```python
# Hypothesis: head-input arrays expose the pre-w2 activation needed for a direct closed-form repair.
print("keys",meta.keys())
for k,v in meta.items():
    if k!="rows": print(k,str(v)[:10000])
print("files",os.listdir(fdir))
for f in fdir.glob("*.npy"):
    a=np.load(f,mmap_mode="r");print(f.name,a.shape)
```
