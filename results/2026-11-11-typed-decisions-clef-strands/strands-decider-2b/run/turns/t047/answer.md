Editing only the dominant dimensions produced no net gain. I’ll now test the complementary edit on the other 2,038 dimensions. I’ll also correct the diagnostic calculations to remove option-common logit offsets and apply the question temperatures.

```python
# Hypothesis: low-energy option-state dimensions are sufficient for nearly all of the validated repair's benefit.
def eval_cached(dd, logits, reference=None):
    mask = dd["mask"]
    pp = (logits/dd["temp"][:, None]).masked_fill(~mask, -torch.inf).softmax(-1)
    ok = (pp.argmax(-1) == dd["gold"].argmax(-1)).float()
    ref = dd["L"] if reference is None else reference
    ok0 = (ref.masked_fill(~mask,-torch.inf).argmax(-1) == dd["gold"].argmax(-1)).float()
    eid_order = list(dict.fromkeys(dd["ids"]))
    item_scores, item_delta = [], []
    for eid in eid_order:
        idx = [i for i, v in enumerate(dd["ids"]) if v == eid]
        item_scores.append(ok[idx].mean().item())
        item_delta.append((ok[idx]-ok0[idx]).mean().item())
    item_scores, item_delta = np.array(item_scores), np.array(item_delta)
    return {"accuracy": float(item_scores.mean()), "gain": float(item_delta.mean()),
            "se": float(item_delta.std(ddof=1)/np.sqrt(len(item_delta))),
            "brier": float(((pp-dd["gold"]).square().sum(-1)).mean())}

def valid_center(logits, mask):
    return (logits - (logits*mask).sum(-1,keepdim=True)/mask.sum(-1,keepdim=True))*mask

print("Correct temperature-scaled Brier:", eval_cached(dev, dev["L"]+Cdev_ref))
for nm, vc in [("top10",v_outlier),("other2038",v_lowenergy),("full",v_ref)]:
    cc = valid_center(correction(dev,vc),dev["mask"])/dev["temp"][:,None]
    print(nm,"RMS within-question log-odds change",float(cc[dev["mask"]].square().mean().sqrt()))
save_edit("k.weight", K0+torch.outer(u,v_lowenergy))
with open(WORK/"notes.md","a") as f:
    f.write("\nPass 16 actual top10-only ablation: development .577778 -> .577778, gain 0, SE .003860; "
            "6 items improved and 6 worsened. Confirms cached prediction.\n")
    f.write(f"Corrected best2 temperature-aware dev statistics: {eval_cached(dev,dev['L']+Cdev_ref)}\n")
    f.write("\n## Before pass 17\nExact edit: k.weight = K0 + outer(u,v_other2038), "
            "v_other2038 equals validated v_ref with the ten highest-energy option dimensions zeroed. "
            "All other tensors original. Paired DECIDE 180 development cases in paired_dev.json. "
            "Prediction .771111 accuracy versus .772222 full repair. Diagnostic only; best2 unchanged on disk.\n")
```
