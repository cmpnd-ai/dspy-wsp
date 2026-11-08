Unit strength should correct more mistakes, while excessive strength should overshoot. I’ll measure that explicitly at 1.5×, and inspect whether the correction changes the same option token differently across questions—evidence against a disguised global bias or temperature adjustment.
```python
# Hypothesis: the correction is state/question-specific and has a bounded dose response, rather than uniformly favoring an option token.
r12=read_arrays(12)
print("Actual unit-strength metrics:",metrics_array(dev,r12["p"]))
print("Unit prediction probability max error:",np.max(np.abs(r12["p"]-dose_predictions[1.0][1])))
baseok=dev["p"].argmax(1)==dev["gold"].argmax(1)
newok=r12["p"].argmax(1)==dev["gold"].argmax(1)
dl=r12["l"]-dev["l"]
dl=dl-np.sum(dl*dev["mask"],axis=1)[:,None]/dev["mask"].sum(axis=1)[:,None]
for key in [("agent_trace_observability","action"),("agent_trace_observability","outcome"),
            ("customer_service","action"),("customer_service","category"),
            ("security_incidents","credential_compromise")]:
    ix=np.array([k==key for k in dev["keys"]])
    row=dev["meta"]["rows"][np.flatnonzero(ix)[0]]
    print(key,"mean centered logit shifts",dict(zip(row["options"],np.mean(dl[ix],axis=0)[:len(row["options"])].round(3))))
    print(" fixed/broken:",int((ix & ~baseok & newok).sum()),int((ix & baseok & ~newok).sum()))
fixix=[i for i,k in enumerate(dev["keys"]) if k==("customer_service","action") and not baseok[i] and newok[i]]
for i in fixix[:2]:
    eid=dev["row_ids"][i]
    print("Corrected",eid,"state",json.dumps(by_id[eid]["inputs"]["state"])[:1200])
    print("Original",dev["p"][i].round(3),"edited",r12["p"][i].round(3),"gold",dev["gold"][i])
install_w2(1.5)
note("Pass 12 actual unit-strength dev metrics: "+str(metrics_array(dev,r12["p"]))+
     ". Actual/predicted probabilities agree to <0.0001. Contrasts differ by workflow/question: not a temperature or global option bias.\n\nPlanned pass 13: READ edited same 180 dev records, candidate_dev.json. Exact edit W2_original + 1.5*delta_w2 (ridge=1, rank<=9). Expected development accuracy 0.710, below unit strength due to overshoot. All other weights original.")
```
