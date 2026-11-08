Removing the state-varying component dropped accuracy from 65.02% to 49.69% on the same 450 records. I’ll restore the frozen edit and prepare matched-state tests to distinguish genuine cue sensitivity from changes in answer frequency.
I’ll restore the frozen edit, quantify the full-versus-prior ablation, and check the exact request format before constructing the final cue tests.
```python
# Hypothesis: matched-state tests can separate improved cue sensitivity from the large correction to fixed answer preferences.
prior_records21 = [json.loads(line) for line in (work/"gen_21_edited.jsonl").read_text().splitlines()]
prior_by_id21 = {r["id"]:r for r in prior_records21}
full_minus_prior = np.array([final_by_id[i]["score"]-prior_by_id21[i]["score"] for i in ablation_ids])
ablation_summary = {"gain":float(full_minus_prior.mean()),
                    "se":float(full_minus_prior.std(ddof=1)/np.sqrt(len(full_minus_prior))),
                    "improved":int((full_minus_prior>0).sum()),
                    "worsened":int((full_minus_prior<0).sum())}
print("Full vs prior-only on450 fitted records:",ablation_summary)
note("\nForward21 prior-only ablation450: original .363111, prior-only .496889, paired gain .133778 SE .012328. Full frozen best4 on same450 .650222. Full minus prior-only "+str(ablation_summary)+
     ". Fitted-record causal diagnostic; cross-fit full-minus-prior independently .070889 SE .008307. Restore best4 now.")
save_edit("scalar.weight",frozen_weight)
print("Existing synthetic cue request example:",json.dumps(cue_requests[0],ensure_ascii=False)[:2400])
print("Existing manifest example:",str(cue_info[0])[:1000])
print("Original call request example:",str(baseline[0]["calls"][0])[:2200])
print("Frozen digest:",hashlib.sha256(load("scalar.weight").numpy().tobytes()).hexdigest())
```
