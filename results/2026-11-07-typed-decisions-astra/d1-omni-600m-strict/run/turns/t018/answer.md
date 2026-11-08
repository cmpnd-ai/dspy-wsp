The 25% attenuation changed no answers and mostly rescaled logits, so it does not support the proposed repair. I’ll test a stronger, 85% attenuation once to check whether the mechanism is genuinely insensitive or simply needs a larger intervention.
I’ll replace the exploratory edit with the stronger two-mode attenuation and log it. No other weights will change.
```python
# Hypothesis: if the dominant MLP modes causally obscure state evidence, a strong selective attenuation should alter rankings and deadline sensitivity.
note("Pass 4 result: .25 attenuation of first-head-MLP top two modes gave exactly zero accuracy/ranking gain on discovery A (.558 unchanged). Logit correlation .999872; best scalar .97811; non-scalar residual 1.60%. Invoice urgency margin near/later unchanged (-2.826/-2.932). This strength is rejected as effectively a confidence rescaling.")
mode_strength=.85
save_edit(mode_name,head_weight_cache[mode_name]-mode_strength*modes02)
note("## Pass 5 planned: head-mode attenuation dose response, strength .85\nExact edit: head.head.layers.0.linear2.weight = original - .85 * sum(top two SVD modes); other weights original. "+json.dumps(edit_manifest())+
     "\nRead edited on discovery A, same 200 cases/1000 questions in mode_read.json. Compare to strengths 0 and .25; retain only if non-global useful ranking changes occur.")
print(edit_manifest())
```
