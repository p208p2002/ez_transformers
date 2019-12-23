<p align="center">
    <br>
    <img src="https://github.com/p208p2002/ez-transformers/blob/master/logo.png?raw=true" width="400"/>
    <br>
    Eazy and useful tools for PyTorch and PyTorch-Transformers
<p>

## Document
### General
#### Usage
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

### data_model
#### Usage
`from eztransfomers.data_model import *`
```
class BertDataModel()
def __init__(self,tokenizer)
def toBertIds(self,input_a,input_b = None)
def add(self, label, input_a, input_b = None)
```

### train_model
#### Usage
`from eztransfomers.train_model import *`
```
class TrainManager()
def __init__(self,
            model,
            optimizer,
            device = 'cpu', # cpu or cuda
            epoch=3,
            learning_rate=5e-6,
            log_interval = 50,
            save_step_interval = 1000
        )
def train(self,train_dataloader,test_dataloader = None)
```
