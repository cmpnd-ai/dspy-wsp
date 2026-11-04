I’ll test whether suppressing the block-3 channel 382 changes the failure pattern. The edited and original models will receive the same four problems with the same sampling seeds and a 2,048-token cap. This is the causal test for the candidate; the edit is only one down-projection column. 

```forward
{"type":"generate","model":"both","problems":["train_5238","train_912","train_204","train_6074"],"n":1,"max_tokens":2048}
```