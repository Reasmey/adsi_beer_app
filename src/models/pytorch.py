import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np

class PytorchRegression(nn.Module):
    def __init__(self, num_features):
        super(PytorchRegression, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 128)
        self.layer_out = nn.Linear(128, 1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)))
        x = self.layer_out(x)
        return (x)
    
# get device   
def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu') # don't have GPU 
    return device

class PytorchDataset(Dataset):
    """
    Pytorch dataset
    ...

    Attributes
    ----------
    X_tensor : Pytorch tensor
        Features tensor
    y_tensor : Pytorch tensor
        Target tensor

    Methods
    -------
    __getitem__(index)
        Return features and target for a given index
    __len__
        Return the number of observations
    to_tensor(data)
        Convert Pandas Series to Pytorch tensor
    """
        
    def __init__(self, X, y):
        self.X_tensor = self.to_tensor(X)
        self.y_tensor = self.to_tensor(y)
    
    def __getitem__(self, index):
        return self.X_tensor[index], self.y_tensor[index]
        
    def __len__ (self):
        return len(self.X_tensor)
    
    def to_tensor(self, data):
        return torch.Tensor(np.array(data))
    
# Solution:
class PytorchBinary(nn.Module):
    def __init__(self, num_features):
        super(PytorchBinary, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 256)
        self.layer_out = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)))
        x = self.sigmoid(self.layer_out(x))
        return (x)
    
    
# Solution:
class PytorchMultiClass(nn.Module):
    def __init__(self, num_features):
        super(PytorchMultiClass, self).__init__()
        
        self.layer_1 = nn.Linear(num_features, 32)
        self.layer_out = nn.Linear(32, 4)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = F.dropout(F.relu(self.layer_1(x)), training=self.training)
        x = self.layer_out(x)
        return self.softmax(x)
    
# Dataset creator for embedding
class EmbeddingDataset(Dataset):
    def __init__(self, data, targets=None,
                 is_train=True, cat_cols_idx=None,
                 cont_cols_idx=None):
        self.data = data
        self.targets = targets
        self.is_train = is_train
        self.cat_cols_idx = cat_cols_idx
        self.cont_cols_idx = cont_cols_idx
    
    def __getitem__(self, idx):
        row = self.data[idx].astype('float32')
        
        data_cat = []
        data_cont = []
        
        result = None
        
        if self.cat_cols_idx:
            data_cat = torch.tensor(row[self.cat_cols_idx])
            
        if self.cont_cols_idx:
            data_cont = torch.tensor(row[self.cont_cols_idx])
                
        data = [data_cat, data_cont]
                
        if self.is_train:
            result = {'data': data,
                      'target': torch.tensor(self.targets[idx])}
        else:
            result = {'data': data}
            
        return result
            
    
    def __len__(self):
        return(len(self.data))
    
