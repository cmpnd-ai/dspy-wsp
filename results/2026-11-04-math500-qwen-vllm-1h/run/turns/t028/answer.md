I’ll test whether fully suppressing these two candidate output directions changes the long-form failure. The paired run uses the same four problems and a 2,048-token cap, allowing direct comparison with the earlier original baseline. The current edit only zeros block-23 down-projection columns 9 and 46. 

```forward
{"type":"generate","model":"both","problems":["train_5238","train_912","train_204","train_6074"],"n":1,"max_tokens":2048}
```