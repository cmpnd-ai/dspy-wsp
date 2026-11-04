I’ll compare the probe edit against the original on the same four problems, using identical sampling seeds and a 512-token cap. The paired comparison will show whether suppressing these two block-23 channels changes answer completion or score in the predicted direction. The edit under test is limited to block-23 down-projection columns 9 and 46, each halved. 

```forward
{"type":"generate","model":"both","problems":["train_5238","train_912","train_204","train_6074"],"n":1,"max_tokens":512}
```