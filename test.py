from eztransfomers import *
from eztransfomers.data_model import *
from eztransfomers.train_model import *
import unittest
from transformers import AlbertConfig, AlbertForSequenceClassification, BertTokenizer,AdamW
import torch

class Test_Core(unittest.TestCase):
    def test_makeTorchDataset(self):
        f1 = [[1,2,3],[4,5,6],[7,8,9]]
        f2 = [[1,2,3],[4,5,6],[7,8,9]]
        f3 = [[1,2,3],[4,5,6],[7,8,9]]
        dataset = makeTorchDataset(f1,f2,f3)
    
    def test_splitDataset(self):
        feature = [[1,2,3],[4,5,6],[7,8,9]]
        dataset = makeTorchDataset(feature,feature,feature,feature)
        train_dataset,test_dataset = splitDataset(dataset,split_rate=0.5)
    
    def test_makeTorchDataLoader(self):
        feature = [[1,2,3],[4,5,6],[7,8,9]]
        dataset = makeTorchDataset(feature,feature,feature,feature)
        train_dataset,test_dataset = splitDataset(dataset,split_rate=0.5)
        makeTorchDataLoader(train_dataset,batch_size = 4,shuffle = True)
        makeTorchDataLoader(test_dataset,batch_size = 2,shuffle = False)

class Test_DataModel(unittest.TestCase):
    def test_BertDataModel(self):
        tokenizer = BertTokenizer.from_pretrained('albert_tiny/vocab.txt')
        model_config = AlbertConfig.from_json_file('albert_tiny/albert_config_tiny.json')
        model = AlbertForSequenceClassification.from_pretrained('albert_tiny/albert_tiny_torch.bin',config = model_config)
        bertDataModel = BertDataModel(tokenizer)
        bertDataModel.toBertIds('你好嗎')
        bertDataModel.toBertIds('你好嗎','我很好')
        bertDataModel.add(input_a='電影很好看哦',label=1)
        bertDataModel.add(input_a='今天的電影如何?',input_b='超級難看',label=0)
        assert len(bertDataModel.features) == 2

class Test_TrainModel(unittest.TestCase):
    def test_TrainManager(self):
        # model
        tokenizer = BertTokenizer.from_pretrained('albert_tiny/vocab.txt')
        model_config = AlbertConfig.from_json_file('albert_tiny/albert_config_tiny.json')
        model = AlbertForSequenceClassification.from_pretrained('albert_tiny/albert_tiny_torch.bin',config = model_config)
        
        #
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,lr=5e-6, eps=1e-8)
        
        # dataloader
        feature = [[1,2,3],[4,5,6]]
        label = [[1],[0]]
        dataset = makeTorchDataset(feature,label)
        # train_dataset,test_dataset = splitDataset(dataset,split_rate=0.5)
        # train_dataloader = makeTorchDataLoader(train_dataset,batch_size = 4,shuffle = True)
        # test_dataloader =makeTorchDataLoader(test_dataset,batch_size = 2,shuffle = False)

        #
        # tm = TrainManager(model=model, optimizer=optimizer)
        # tm.train(train_dataloader=train_dataloader, test_dataloader=test_dataloader)

if __name__ == "__main__":
    unittest.main()
