The added query-context paths are not supported by the test comparison, so I’m reverting them. I’ll examine fit and development errors from the simpler repair to distinguish a remaining readout problem from evidence the backbone may not encode.


```python
# Hypothesis: the simpler rank-one option-evidence readout generalizes better than extra context-conditioned paths.
a13 = {r["id"]: r["score"] for r in r13}
a15 = {r["id"]: r["score"] for r in r15}
inc15 = np.array([a15[k] - a13[k] for k in a13])
save_edit("k.weight", torch.from_numpy(np.load(WORK / "best_2/edited/k.weight.npy")).float())
print("Rank5 minus best2 on test:", float(inc15.mean()), "paired SE:", float(inc15.std(ddof=1)/np.sqrt(len(inc15))))
print("Edited:", edited())
print("Helper functions:", [name for name, obj in list(globals().items()) if inspect.isfunction(obj)])
print("Data samples:")
for key in ("ids", "group", "meta", "temp", "g"):
    obj = dev[key]
    print(key, obj[:2] if not isinstance(obj, torch.Tensor) else (tuple(obj.shape), obj[:2].tolist()))
print("Ranges L, gold:", float(dev["L"][dev["mask"]].min()), float(dev["L"][dev["mask"]].max()),
      dev["gold"][:2].tolist())
print("Best2 derivation keys:", list(torch.load(WORK/"best_2/weights_and_direction.pt", weights_only=False)))
ex_by_id = {e["id"]: e for e in examples}
for data_name, dd, CC in [("fit", fit, Cfit_ref), ("dev", dev, Cdev_ref)]:
    pp0 = dd["L"].masked_fill(~dd["mask"], -torch.inf).argmax(-1)
    pp1 = (dd["L"]+CC).masked_fill(~dd["mask"], -torch.inf).argmax(-1)
    tt = dd["gold"].argmax(-1)
    print(data_name, "overall accuracy", (pp0==tt).float().mean().item(), (pp1==tt).float().mean().item())
    if data_name == "dev":
        for gg in sorted(set(dd["group"])):
            idx = torch.tensor([i for i, v in enumerate(dd["group"]) if v == gg])
            print(gg, "base/refined", round((pp0[idx]==tt[idx]).float().mean().item(),3),
                  round((pp1[idx]==tt[idx]).float().mean().item(),3),
                  "wrong transitions", collections.Counter(zip(pp1[idx][pp1[idx]!=tt[idx]].tolist(),tt[idx][pp1[idx]!=tt[idx]].tolist())).most_common(3))
with open(WORK/"notes.md", "a") as f:
    f.write("\nPass 15 rank<=5 key practice-test: original .591111 -> edited .744444, gain .153333, SE .021205. "
            f"Increment over best2 {inc15.mean():+.6f}, paired SE {inc15.std(ddof=1)/np.sqrt(len(inc15)):.6f}. "
            "Extra context paths rejected; restored exact best_2/edited/k.weight.npy. "
            "Current edit is again validated rank-one K0+outer(u,v_ref), all other tensors original.\n")
```
