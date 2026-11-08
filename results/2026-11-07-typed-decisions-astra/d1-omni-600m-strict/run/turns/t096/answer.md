The gate test confirms that the violation effect can be localized, but localization alone did not improve accuracy. I’ll restore the validated weights and audit them on matched counterfactuals, changing only payment deadlines, agent violations or security impact cues.
```python
# Hypothesis: the retained repair responds to the intended state cues rather than merely shifting every answer distribution.
for name in list(edited()):
    checkpoint=work/"best_2"/(name+".npy")
    if checkpoint.exists():
        save_edit(name,torch.from_numpy(np.load(checkpoint)).clone())
    else:
        drop_edit(name)
audit_requests23=[]
audit_map23=[]
for family,filename,passno,nrequests in [
    ("payment","deadline_counterfactuals.json",6,64),
    ("violation","agent_counterfactuals.json",9,64),
    ("security","security_counterfactuals.json",14,64)]:
    source23=json.loads((work/filename).read_text())
    for source_index23,request23 in enumerate(source23["requests"][:nrequests]):
        audit_map23.append(dict(request=len(audit_requests23),family=family,
                                original_pass=passno,original_request=source_index23))
        audit_requests23.append(request23)
audit23={"type":"read","model":"edited","requests":audit_requests23}
(work/"final_cue_audit.json").write_text(json.dumps(audit23))
(work/"final_cue_audit_map.json").write_text(json.dumps(audit_map23))
print("REQUESTS",len(audit_requests23),"QUESTIONS",sum(len(r["questions"]) for r in audit_requests23))
print("MANIFEST",edit_manifest())
with open(work/"notes.md","a") as f:
    f.write("\n## Pass 23 planned: final matched-cue audit\n"
            "Exact edit restored best_2 (validated pass21), "+json.dumps(edit_manifest())+"\n"
            "READ edited 192 counterfactual requests/960 questions: all64 payment requests from pass6, "
            "first64 agent requests from pass9, first64 security requests from pass14. "
            "Exact inputs and mapping in final_cue_audit.json/final_cue_audit_map.json. "
            "Compare with original cached reads, no metric fitting or strength selection. "
            "Prediction: stronger selective responses to deadline, violation and asset/privilege cues.\n")
```
