from eztorch import *
import unittest


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

if __name__ == "__main__":
    unittest.main()