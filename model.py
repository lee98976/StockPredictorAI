import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import random
from datetime import datetime
import numpy

class StockModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(StockModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 258)
        self.fc2 = nn.Linear(258, 2)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()

        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = torch.relu(self.fc1(out))
        out = self.fc2(out)
        # out = nn.Softmax(1)(out)
        return out