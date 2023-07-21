"""
Author: Liu
Time: 2023.06.14
"""
from torch.utils.data import Dataset


class GaitDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index: int):
        dataRet = self.data[index]
        labelRet = self.label[index]
        return dataRet, labelRet

    def __len__(self):
        oneVariable = self.data.shape
        return oneVariable[0]
