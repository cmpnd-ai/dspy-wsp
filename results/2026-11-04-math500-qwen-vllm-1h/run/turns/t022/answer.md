I’ll rerun four diagnostic problems with a 512-token cap and collect residual-stream and MLP-input activations across the model. The shorter generations let us compare the onset of the self-checking loop without spending the full token budget, and the activations may help localize it. No weights have been changed.  
```forward
{"type":"generate","model":"original","problems":["train_5238","train_912","train_204","train_6074"],"n":1,"max_tokens":512,"activations":{"layers":[0,4,8,12,16,20,24],"mlp_layers":[4,8,12,16,20,23]}}
```