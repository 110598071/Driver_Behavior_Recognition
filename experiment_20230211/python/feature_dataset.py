from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

class FeatureDataset(Dataset):
    def __init__(self, data_list, label_list):
        self.data_list = torch.from_numpy(data_list.astype(np.float32))
        self.label_list = torch.from_numpy(label_list.astype(np.float32))
        self.len = self.data_list.shape[0]
       
    def __getitem__(self, index):
        return self.data_list[index], self.label_list[index]
   
    def __len__(self):
        return self.len