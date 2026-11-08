Asset criticality alone does not explain every failure: access to production and secrets should also distinguish high from medium severity. I’ll create a factorial counterfactual test varying those two cues independently, while holding each seed’s alert evidence fixed.
```python
# Hypothesis: two distinct underread impact cues—critical assets and production/secret privileges—can be isolated with matched state interventions.
security_seed_ids = []
seen_rules = set()
for eid in security_a_ids:
    rule = by_id[eid]["inputs"]["state"]["alert"]["rule"]
    if rule not in seen_rules:
        security_seed_ids.append(eid)
        seen_rules.add(rule)
for eid in security_a_ids:
    if eid not in security_seed_ids and len(security_seed_ids)<20:
        security_seed_ids.append(eid)
print("Seeds:",len(security_seed_ids),collections.Counter(by_id[e]["inputs"]["state"]["alert"]["rule"] for e in security_seed_ids))
requests_a_list = json.loads((WORK/"fwd_2"/"requests.json").read_text())
requests_a_byid = {request_to_id[canonical(r)]:r for r in requests_a_list}
security_cf_requests, security_cf_tags = [], []
for eid in security_seed_ids:
    for criticality in ["low","medium","high","crown_jewel"]:
        for privileged in [False,True]:
            req = copy.deepcopy(requests_a_byid[eid])
            st = req["state"]["inputs"]["state"]
            st["context"]["asset_criticality"] = criticality
            st["principal"]["privileges"] = ["deploy:production","secrets:read"] if privileged else ["repo:read"]
            security_cf_requests.append(req)
            security_cf_tags.append({"seed":eid,"criticality":criticality,"privileged":privileged})
(WORK/"security_counterfactuals.json").write_text(json.dumps({"type":"read","model":"original","requests":security_cf_requests}))
(WORK/"security_counterfactual_tags.json").write_text(json.dumps(security_cf_tags))
note("\n## Pass 14 planned: causal security impact cues\nCurrent edits unchanged best_1 payment4_violation2, but READ ORIGINAL. Original reads 20 discovery-A security seeds x8 asset-criticality/privilege variants, 160 requests/800 questions (security_counterfactuals.json). Asset low/medium/high/crown_jewel crossed with repo:read vs deploy:production,secrets:read; alert evidence otherwise held fixed. Test whether severity-contrast representations track these explicit state cues while raw logits default to medium.")
print("Counterfactual requests:",len(security_cf_requests),"questions",5*len(security_cf_requests))
```
