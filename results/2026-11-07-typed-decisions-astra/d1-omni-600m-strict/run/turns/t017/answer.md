I’ll compare the edited and original logits, including deadline sensitivity and question-level accuracy. If the two-mode change only flattens distributions without correcting choices, I’ll reject it.
```python
# Hypothesis: a useful mode attenuation changes state-dependent rankings, not just the scale of all logits.
arr4,rowinfo4,meta4=load_read(4,read_ids_a)
def read_summary(arr,info,base_arr=None,base_info=None):
    acc=float(np.mean([r["ok"] for r in info]))
    bi={ (r["id"],r["q"]):r for r in base_info } if base_info else {}
    print("Overall accuracy",round(acc,4))
    item_delta=collections.defaultdict(list)
    for key in sorted(groups):
        rs=[r for r in info if (r["wf"],r["q"])==key]
        if bi:
            dd=[float(r["ok"])-float(bi[r["id"],r["q"]]["ok"]) for r in rs]
            for r,d in zip(rs,dd): item_delta[r["id"]].append(d)
            print(key, "acc",round(float(np.mean([r["ok"] for r in rs])),3), "delta",round(float(np.mean(dd)),3))
    if bi:
        ds=np.array([np.mean(v) for v in item_delta.values()])
        print("Paired item delta",float(ds.mean()),"SE",float(ds.std(ddof=1)/np.sqrt(len(ds))),"improved/worse",(int((ds>0).sum()),int((ds<0).sum())))
def deadline_summary(arr,info):
    rr=[r for r in info if r["wf"]=="invoice_processing" and r["q"]=="urgency"]
    for flag in [False,True]:
        rs=[r for r in rr if (by_id[r["id"]]["inputs"]["state"]["payment"]["days_until_due"]<=3)==flag]
        print("near",flag, "pred",collections.Counter(r["opts"][r["pi"]] for r in rs),
              "logit elevated-routine",round(float(np.mean([arr["option_logits"][r["i"],2]-arr["option_logits"][r["i"],1] for r in rs])),3))
read_summary(arr4,rowinfo4,arr2,rowinfo2)
deadline_summary(arr4,rowinfo4)
L2=np.asarray(arr2["option_logits"])
L4=np.asarray(arr4["option_logits"])
def centered_options(L,info):
    return np.concatenate([L[r["i"],:len(r["opts"])]-np.mean(L[r["i"],:len(r["opts"])]) for r in info])
lc2,lc4=centered_options(L2,rowinfo2),centered_options(L4,rowinfo4)
print("Logit correlation",np.corrcoef(lc2,lc4)[0,1],"best scalar",float(lc2@lc4/(lc2@lc2)),
      "non-scale residual fraction",float(np.linalg.norm(lc4-lc2*(lc2@lc4/(lc2@lc2)))/np.linalg.norm(lc4)))
```
