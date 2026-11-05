The set-attention correction varies by only about 0.14 logits within a typical choice question, while correcting the scalar projection improves internal holdout accuracy from roughly 36% to 52%. I’ll test a wider closed-form regularization range and inspect which errors it fixes before selecting an intervention strength.

```python
# Hypothesis: low-variance EOS directions carry useful state evidence that the original scalar head underweights.
candidate_dw={}
for lam in [0.0001,0.001,0.01,0.03,0.1,0.3,1]:
    dw=U@((U.T@rhs)/(evals+lam))
    candidate_dw[lam]=dw
    print("lambda",lam,"norm",round(np.linalg.norm(dw),2),
          "internal holdout",eval_dw(dw,designs[2][2]))
print("Model files:",[p.name for p in pathlib.Path("/Users/dbreunig/.cache/huggingface/hub/models--C-Tianyu--NanoJev/snapshots/047b927b30882a1138fc504821b82ac145a4b81a").iterdir()])
dw=candidate_dw[.03]
for strength in [0,.25,.5,.75,1,1.25,1.5]:
    print("strength",strength,eval_dw(dw,designs[2][2],strength))
for j in [0,1,2,3,4]:
    h,z,p,r=designs[0][2][j]
    print(batches[0]["ids"][j],r["question"],"gold",r["options"][p.argmax()],
          "before",r["options"][z.argmax()],"after",r["options"][(z+h@dw).argmax()])
```
