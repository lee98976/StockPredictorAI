import os
import random
import time
from datetime import datetime

import discord
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
from torchinfo import summary

torch.set_default_device('cpu')

#DO NOT USE REAL MONEY WITH THIS: THIS IS A PAPER ACCOUNT
api_key = 'PKZ5HQ89HCEAW6H0QSPI'
api_secret = '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L'

trading_client = TradingClient(api_key, api_secret, paper=True)
trading_stream_client = TradingStream(api_key, api_secret, paper=True)
stream_client = StockDataStream(api_key, api_secret)
data_client = StockHistoricalDataClient(api_key, api_secret)

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

def loadData():
    if input("Do you want to create a new test set or not? Y/N: ") == "Y":
        number = int(input("How many test cases?: "))
        test = StockData(False, number)
        data = test.inputData
        labels = test.answer

        with open('Dataset/stockData.json', 'w') as json_file:
            json.dump(data, json_file)

        with open('Dataset/labels.json', 'w') as json_file:
            json.dump(labels, json_file)
    else:
        with open('Dataset/stockData.json', 'r') as json_file:
            data = json.load(json_file)

        with open('Dataset/labels.json', 'r') as json_file:
            labels = json.load(json_file)
    return data, labels

# data, labels = loadData()

# print(data)
# print(labels)

def loadModel():
    global model
    try:
        print("Attempting to load model...")
        model1 = torch.jit.load('ModelSave/model_scripted.pt')
        model = model1
        model
        print("Load Successful!")
    except:
        print("No file detected.")

def trainer(train_loader):
    global model
    loadModel()
    
    epoches = int(input("How many epoches do you want to train for?: "))
    for epoch in range(epoches):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        
        print("Epoch: " + str(epoch+1) + "/" + str(epoches) + ", Loss: " + str(running_loss/len(train_loader)))

    print("The model has finished cooking.")

    model_scripted = torch.jit.script(model) # Export to TorchScript
    model_scripted.save('ModelSave/model_scripted.pt') # Save
    print("Saving...")

def evaluation(train_loader):
    global model
    loadModel()
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        running_loss = 0.0
        for inputs, labels in train_loader: #Create test instead
            model.eval()
            output = model(inputs)
            test_loss = criterion(output, labels)
            running_loss += test_loss
    
    print("Accuracy: " + str(1 - running_loss/len(train_loader)))

def requestData(self, stock : str, timeframe, start : datetime, end : datetime):
        request_params = StockBarsRequest(
            symbol_or_symbols = [stock],
            timeframe = timeframe,
            start=start,
            end=end,
        )
        bars = data_client.get_stock_bars(request_params)
        return bars

def prediction():
    global model
    loadModel()
    model.eval()

    with torch.no_grad():
        test = StockData(False, 1)
        data = test.inputData
        labels = test.answer
        dataset = TensorDataset(data, labels)
        train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
        for inputs, labels in train_loader:
            output = model(inputs)
            print(output)
        

def main():
    data, labels = loadData()
    data, labels = torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)
    dataset = TensorDataset(data, labels)
    train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
    model = StockModel(4, 1024, 3)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)

    # summary(model, (100, 4, 4), device="cpu")
    summary(model, (100, 4, 4))

    trainer(train_loader)
    evaluation(train_loader)
main()

# Discord Bot!

# intents = discord.Intents.default()
# intents.message_content = True

# client = discord.Client(intents=intents)

# @client.event
# async def on_ready():
#     print(f'Bot logged in as {client.user}.')

# @client.event
# async def on_message(message):
#     if message.author == client.user or message.author != "lee98976":
#         return

#     if message.content.startswith('$Stock '):
#         try:
#             if message.content.split()[1] == "NVDA":
#                 await message.channel.send('Prediction: ')
#         except:
#             pass

# client.run('')
# print("Bot logged out.")

'''
To-Do:
1. Organize everything into functions
2. Load from JSON file #
3. Improve model #
4. Make Discord Bot #
5. Use it on Alpaca
6. Make model save
'''
