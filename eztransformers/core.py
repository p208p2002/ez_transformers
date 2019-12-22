import torch
from torch.utils.data import TensorDataset, DataLoader, random_split

def makeTorchDataset(*features):
    tensor_features = []
    for feature in features:
        tensor_feature = torch.tensor([f for f in features])
        tensor_features.append(tensor_feature)
    return TensorDataset(*tensor_features)

def splitDataset(full_dataset,split_rate = 0.8):
    train_size = int(split_rate * len(full_dataset))
    test_size = len(full_dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
    return train_dataset, test_dataset

def makeTorchDataLoader(torch_dataset,**options):
    #options: batch_size=int,shuffle=bool
    return DataLoader(torch_dataset,**options)


if __name__ == "__main__":
    f1 = [[1,2,3],[4,5,6],[7,8,9]]
    f2 = [[1,2,3],[4,5,6],[7,8,9]]
    f3 = [[1,2,3],[4,5,6],[7,8,9]]

    makeTrochDataset(f1,f2,f3)
