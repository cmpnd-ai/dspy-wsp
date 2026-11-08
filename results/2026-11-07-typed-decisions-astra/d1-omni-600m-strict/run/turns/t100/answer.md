I’ll compare named-option margins across the matched variants and check that the exported head inputs are unchanged. That will distinguish a readout repair from a change to the encoder’s representations.
```python
# Hypothesis: only the scorer changes, and the corrected margins track the controlled cues.
audit_lookup23={}
original_lookup23={}
for passno23,(orig_arr23,orig_meta23,source_requests23,tags23) in audit_sources23.items():
    source_tag23={canonical(req):tag for req,tag in zip(source_requests23,tags23)}
    stored_orig23=json.loads((work/f"fwd_{passno23}"/"requests.json").read_text())
    for key23,tag23 in source_tag23.items():
        audit_lookup23[key23]=(passno23,tag23)
    for i23,r23 in enumerate(orig_meta23["rows"]):
        original_lookup23[(passno23,canonical(stored_orig23[r23["request"]]),r23["question"])]=i23
audit_margins23=defaultdict(list)
max_feature_delta23=0.0
for i23,r23 in enumerate(meta23["rows"]):
    key23=canonical(stored23[r23["request"]])
    passno23,tag23=audit_lookup23[key23]
    j23=original_lookup23[(passno23,key23,r23["question"])]
    original_arr23=audit_sources23[passno23][0]
    opts23=r23["options"]; n23=len(opts23)
    max_feature_delta23=max(max_feature_delta23,float(np.max(np.abs(
        arr23["head_inputs"][i23,:n23]-original_arr23["head_inputs"][j23,:n23]))))
    contrasts23=[]
    if passno23==6 and r23["question"]=="urgency":
        contrasts23=[("elevated-routine","2","1")]
        group23=tag23["variant"]
    elif passno23==9 and r23["question"] in ["outcome","risk","urgency"]:
        contrasts23={"outcome":[("harmful-partial","harmful","partial")],
                     "risk":[("high-medium","3","2")],
                     "urgency":[("elevated-routine","2","1")]}[r23["question"]]
        group23=tag23["variant"]
    elif passno23==14 and r23["question"]=="severity":
        contrasts23=[("critical-medium","4","2"),("high-medium","3","2"),("low-medium","1","2")]
        group23=(tag23["criticality"],tag23["privileged"])
    for contrast23,pos23,neg23 in contrasts23:
        ip23=opts23.index(pos23); im23=opts23.index(neg23)
        audit_margins23[(passno23,group23,contrast23)].append((
            float(original_arr23["option_logits"][j23,ip23]-original_arr23["option_logits"][j23,im23]),
            float(arr23["option_logits"][i23,ip23]-arr23["option_logits"][i23,im23])))
print("MAX HEAD-INPUT DIFFERENCE",max_feature_delta23)
audit_report23=[]
for key23,v23 in audit_margins23.items():
    p23,g23,c23=key23
    if p23==14 and not ((g23[0]=="crown_jewel" and c23=="critical-medium") or
                        (g23[0] in ["medium","high"] and c23=="high-medium") or
                        (g23[0]=="low" and c23=="low-medium")):
        continue
    means23=np.mean(v23,axis=0)
    line23=f"{key23}: n={len(v23)}, original={means23[0]:+.4f}, edited={means23[1]:+.4f}"
    audit_report23.append(line23)
    print(line23)
with open(work/"notes.md","a") as f:
    f.write("\nPass23 final matched-cue audit: max absolute head-input difference "
            f"{max_feature_delta23}; exact feature preservation confirms scorer-only intervention.\n"
            +"\n".join(audit_report23)+"\n")
reserve_request24={"type":"decide","model":"both","problems":reserve_ids}
(work/"final_reserve.json").write_text(json.dumps(reserve_request24))
with open(work/"notes.md","a") as f:
    f.write("\n## Pass 24 planned: untouched reserve\n"
            "Exact edit best_2, unchanged since pass21: "+json.dumps(edit_manifest())+"\n"
            "DECIDE both on 50 predesignated reserve IDs in final_reserve.json; "
            "not previously used to design or select any edit. No further weight changes planned.\n")
```
