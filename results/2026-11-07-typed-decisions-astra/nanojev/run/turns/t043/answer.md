The function listing included large default tensors, so it was cut off. I’ll retrieve only the loader and scoring code, plus a few metadata samples, to check alignment without printing model arrays.
```python
# Hypothesis: reproducing the earlier solve exactly will distinguish a data-size benefit from a changed fitting recipe.
for fn_name in ["load_read_dataset", "centered", "metric_logits"]:
    fn = globals()[fn_name]
    try:
        src = inspect.getsource(fn)
    except (OSError, TypeError):
        src = {"names": fn.__code__.co_names, "constants": str(fn.__code__.co_consts)[:1800]}
    print(fn_name, src)
print("Container names:", [(k,type(v).__name__,len(v)) for k,v in list(globals().items())
                           if not k.startswith("_") and isinstance(v,(dict,list))
                           and k not in ["NAMES","SHAPES","head0"]])
print("Row samples:", rows[0], rows[-1])
print("Split sizes:", [(k,len(v),v[:2]) for k,v in splits.items()])
m16 = json.loads((work/"fwd_16"/"meta.json").read_text())
print("Meta16 keys:", list(m16))
for k,v in m16.items():
    if isinstance(v, list):
        print(k, "length",len(v), "first",str(v[0])[:1800])
    else:
        print(k,str(v)[:800])
print("Feature/target samples:", X.shape, G[:2].round(3), mask[:2], counts[:2], RW[:2],
      Tmix[:2].round(3), L0[:2].round(3))
print("Norms:", float(np.linalg.norm(w0)), float(np.linalg.norm(d_mix600)))
```
