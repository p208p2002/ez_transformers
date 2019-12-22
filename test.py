from eztransfomers import *
from eztransfomers.data_model import *
import unittest
from transformers import AlbertConfig, AlbertForSequenceClassification, BertTokenizer
import torch

class TestEZTransformers(unittest.TestCase):
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

class TestEZTransformersDataModel(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
