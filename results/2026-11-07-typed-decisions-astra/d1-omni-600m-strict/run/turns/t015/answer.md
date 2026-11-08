Two singular directions in the first head MLP align strongly with the head’s dominant activation changes. Meanwhile, the deadline contrast replicates, but near-due invoices still trail “routine” by about 2.8 raw logits. I’ll test whether those amplified MLP directions are responsible by reducing only those two modes by 25%.
This is a causal test, not a final edit: all encoder weights and all other head directions will stay unchanged. I expect the intervention to reduce the fixed routine preference relative to state-sensitive evidence; the read pass will show whether it does.
I’ll save the two-mode reduction and record its exact strength and discovery cases before the read pass.
```python
# Hypothesis: excessive amplification by the first head MLP's top two singular modes suppresses deadline-sensitive option distinctions.
import hashlib, shutil
def edit_manifest():
    result=[]
    for n in edited():
        t=load(n).contiguous()
        orig=load_original(n)
        result.append({"name":n,"delta_norm":float((t-orig).norm()),"relative_delta":float((t-orig).norm()/orig.norm()),
                       "sha256":hashlib.sha256(t.numpy().tobytes()).hexdigest()})
    return result
def clear_edits():
    for n in list(edited()):
        drop_edit(n)
mode_name="head.head.layers.0.linear2.weight"
u0,s0,v0=head_svd[mode_name]
modes02=(u0[:,:2]*s0[:2])@v0[:2]
clear_edits()
mode_strength=.25
save_edit(mode_name,head_weight_cache[mode_name]-mode_strength*modes02)
read_mode_req={"type":"read","model":"edited","from":{"pass":1,"ids":read_ids_a},"layers":[16]}
(WORK/"mode_read.json").write_text(json.dumps(read_mode_req))
print("Current edit",edit_manifest())
note("## Pass 4 planned: causal head-mode attenuation, strength .25\nExact edit: head.head.layers.0.linear2.weight = original - .25 * sum(top two SVD modes), no other changes. "+json.dumps(edit_manifest())+
     "\nRead edited model on the same 200 discovery-A cases (mode_read.json), 1000 questions. Predict reduction of fixed routine/benign dominance and relative increase of state signal. This is exploratory; no confirmation examples used.")
```
