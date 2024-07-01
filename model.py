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
        self.fc1 = nn.Linear(hidden_size, 1)
        self.init_weights()  # Initialize weights
        

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)  # Xavier initialization for weights
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)  # Zero initialization for biases
        nn.init.xavier_normal_(self.fc1.weight)  # Xavier initialization for fully connected layer weights
        nn.init.constant_(self.fc1.bias, 0.0)  # Zero initialization for fully connected layer biases

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).requires_grad_()
        out, (h1, c1) = self.lstm(x, (h0.detach(), c0.detach()))
        out = out[:, -1, :]
        out = self.fc1(out)
        return out
