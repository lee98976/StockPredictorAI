import os
import random
import time
from datetime import datetime

import numpy
import torch
import json
import torch.optim as optim
from stockData import StockData
from model import StockModel
import torch.nn as nn
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.live import StockDataStream
from alpaca.data.requests import StockBarsRequest, StockLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, OrderStatus, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrdersRequest, MarketOrderRequest
from alpaca.trading.stream import TradingStream
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset, TensorDataset

#DO NOT USE REAL MONEY WITH THIS: THIS IS A PAPER ACCOUNT
api_key = 'PKZ5HQ89HCEAW6H0QSPI'
api_secret = '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L'

trading_client = TradingClient(api_key, api_secret, paper=True)
trading_stream_client = TradingStream(api_key, api_secret, paper=True)
stream_client = StockDataStream(api_key, api_secret)

account = trading_client.get_account()

def createOrder(stock : str, qty : int, side, time_in_force):
    global trading_client
    orderData = MarketOrderRequest(
        symbol = stock,
        qty = qty,
        side = side,
        time_in_force = time_in_force
    )
    market_order_data = trading_client.submit_order(order_data=orderData)
    return market_order_data

def deleteOrder(stockID : str):
    global trading_client
    trading_client.cancel_order_by_id(stockID)

def deleteAll():
    global trading_client
    trading_client.cancel_orders()

def getOrders(status, side):
    global trading_client
    requestParams = GetOrdersRequest(
        status=status,
        side=side
    )

    orders = trading_client.get_orders(filter=requestParams)
    return orders

def fit_transform(x):
    mu = numpy.mean(x, axis=(0), keepdims=True)
    sd = numpy.std(x, axis=(0), keepdims=True)
    normalized_x = (x - mu)/sd
    return normalized_x


model = StockModel(4, 1024, 3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

test = StockData(100)
data = test.inputData
labels = test.answer

with open('stockData.json', 'w') as json_file:
    json.dump(data, json_file)

with open('labels.json', 'w') as json_file:
    json.dump(labels, json_file)

print(data)
print(labels)

data, labels = torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)

dataset = TensorDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)

for epoch in range(109):
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print("Epoch: " + str(epoch+1) + "/109, Loss: " + str(running_loss/len(train_loader)))

print("COOKED")

model.eval()
correct = 0
total = 0

with torch.no_grad():
    for inputs, labels in train_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)

'''
To-Do:
1. Organize everything into functions
2. Load from JSON file
3. Improve model
4. Make Discord Bot
5. Use it on Alpaca
'''
