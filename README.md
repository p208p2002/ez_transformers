<p align="center">
    <br>
    <img src="https://github.com/p208p2002/ez-transformers/blob/master/logo.png?raw=true" width="400"/>
    <br>
    Eazy and useful tools for PyTorch and PyTorch-Transformers
<p>

## Document
### eztransfomers general api
#### usage
`from eztransfomers import *`
```
def saveModel(model,name)
```
```
def computeAccuracy(y_pred, y_target)
```
```
def makeTorchDataset(*features)
```
```
def splitDataset(full_dataset,split_rate = 0.8)
```
```
def makeTorchDataLoader(torch_dataset,**options)
#options: batch_size=int,shuffle=bool
```
```
def log(*logs)
```
```
def blockPrint()
```
```
def enablePrint()
```
