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
    
# embedding example
class ClassificationEmbdNN(torch.nn.Module):
    
    def __init__(self, emb_dims, no_of_cont=None):
        super(ClassificationEmbdNN, self).__init__()
        
        self.emb_layers = torch.nn.ModuleList([torch.nn.Embedding(x, y)
                                               for x, y in emb_dims])
        
        no_of_embs = sum([y for x, y in emb_dims])
        self.no_of_embs = no_of_embs
        self.emb_dropout = torch.nn.Dropout(0.2)
        
        self.no_of_cont = 0
        if no_of_cont:
            self.no_of_cont = no_of_cont
            self.bn_cont = torch.nn.BatchNorm1d(no_of_cont)
        
        self.fc1 = torch.nn.Linear(in_features=self.no_of_embs + self.no_of_cont, 
                                   out_features=208)
        self.dropout1 = torch.nn.Dropout(0.2)
        self.bn1 = torch.nn.BatchNorm1d(208)
        self.act1 = torch.nn.ReLU()
        
        self.fc2 = torch.nn.Linear(in_features=208, 
                                   out_features=208)
        self.dropout2 = torch.nn.Dropout(0.2)
        self.bn2 = torch.nn.BatchNorm1d(208)
        self.act2 = torch.nn.ReLU()
        
#         self.fc3 = torch.nn.Linear(in_features=256, 
#                                    out_features=64)
#         self.dropout3 = torch.nn.Dropout(0.2)
#         self.bn3 = torch.nn.BatchNorm1d(64)
#         self.act3 = torch.nn.ReLU()
        
        self.fc3 = torch.nn.Linear(in_features=208, 
                                   out_features=104)
        self.act3 = torch.nn.Softmax()
        
    def forward(self, x_cat, x_cont=None):
        if self.no_of_embs != 0:
            x = [emb_layer(x_cat[:, i])
                 for i, emb_layer in enumerate(self.emb_layers)]
        
            x = torch.cat(x, 1)
            x = self.emb_dropout(x)
            
        if self.no_of_cont != 0:
            x_cont = self.bn_cont(x_cont)
            
            if self.no_of_embs != 0:
                x = torch.cat([x, x_cont], 1)
            else:
                x = x_cont
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.bn1(x)
        x = self.act1(x)
        
        x = self.fc2(x)
        x = self.dropout2(x)
        x = self.bn2(x)
        x = self.act2(x)
        
#         x = self.fc3(x)
#         x = self.dropout3(x)
#         x = self.bn3(x)
#         x = self.act3(x)
        
        x = self.fc3(x)
        x = self.act3(x)
        
        return x