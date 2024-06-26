import os
import random
import time
from datetime import datetime
import numpy as np
import torch
import discord
import json
import torch.optim as optim
from stockData import StockData, RealTimeData
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

api_key = 'PKZ5HQ89HCEAW6H0QSPI'
api_secret = '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L'

trading_client = TradingClient(api_key, api_secret, paper=True)
trading_stream_client = TradingStream(api_key, api_secret, paper=True)
stream_client = StockDataStream(api_key, api_secret)
data_client = StockHistoricalDataClient(api_key, api_secret)

account = trading_client.get_account()

def createOrder(stock, qty, side, time_in_force):
    orderData = MarketOrderRequest(
        symbol=stock,
        qty=qty,
        side=side,
        time_in_force=time_in_force
    )
    market_order_data = trading_client.submit_order(order_data=orderData)
    return market_order_data

def normalize_data(data, labels):
    data_scaler = MinMaxScaler(feature_range=(-1, 1))
    labels_scaler = MinMaxScaler(feature_range=(-1, 1))
    
    # Flatten data for scaling
    data = np.array(data).reshape(-1, 365 * 4)
    labels = np.array(labels).reshape(-1, 1)

    if data.shape == (1, 1460):
        data = data.reshape(1460, 1)
    
    # Fit and transform the data
    data = data_scaler.fit_transform(data).reshape(-1, 365, 4)
    labels = labels_scaler.fit_transform(labels)
    
    return data, labels, data_scaler, labels_scaler

def deleteOrder(stockID):
    trading_client.cancel_order_by_id(stockID)

def deleteAll():
    trading_client.cancel_orders()

def getOrders(status, side):
    requestParams = GetOrdersRequest(
        status=status,
        side=side
    )
    orders = trading_client.get_orders(filter=requestParams)
    return orders

def fit_transform(x):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(x)

def loadData():
    if input("Do you want to create a new test set or not? Y/N: ") == "Y":
        number = int(input("How many test cases?: "))
        test = StockData(["NVDA", "AMD", "INTC", "GOOGL", "AMC", "GME", "IBM", "AAPL", "MSFT"], number)
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

def loadModel():
    global model
    try:
        print("Attempting to load model...")
        model1 = torch.jit.load('ModelSave/model_scripted.pt')
        model = model1
        print("Load Successful!")
    except:
        print("No file detected.")

def trainer(train_loader):
    global model
    loadModel()

    epochs = int(input("How many epochs do you want to train for?: "))
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for inputs, labels in train_loader:
            optimizer.zero_grad()

            outputs = model(inputs)
            
            # print(f"Outputs shape: {outputs.shape}")
            # print(f"Labels shape: {labels.shape}")
            
            # print(outputs)
            loss = criterion(outputs, labels)

            print(f"Outputs: {outputs[:5].detach()}")  # Print first few outputs for debugging
            print(f"Labels: {labels[:5]}")  # Print first few labels for debugging
            print(f"Loss: {loss.item()}")  # Print the loss for debugging

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch: {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

    print("The model has finished training.")

    model_scripted = torch.jit.script(model)
    model_scripted.save('ModelSave/model_scripted.pt')
    print("Model saved.")

def evaluation(train_loader):
    global model
    loadModel()
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            test_loss = criterion(outputs, labels)
            running_loss += test_loss.item()

            print(f"Outputs: {outputs[:5]}")
            print(f"Labels: {labels[:5]}")

            # Denormalize for comparison
            denorm_outputs = labels_scaler.inverse_transform(outputs.detach().numpy())
            denorm_labels = labels_scaler.inverse_transform(labels.detach().numpy())

            print(f"Denormalized Outputs: {denorm_outputs[:5]}")
            print(f"Denormalized Labels: {denorm_labels[:5]}")

    print(f"Evaluation Loss: {running_loss/len(train_loader):.4f}")

def prediction(symbol):
    global model
    # loadModel()

    with torch.no_grad():
        model.eval()
        test = RealTimeData(symbol)
        # print(test.inputData, test.answer)
        data = [test.inputData]
        label = [test.answer]
        data, label, data_scaler, labels_scaler = normalize_data(data, label)
        data = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        dataset = TensorDataset(data, label)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)


        for inputs, labels in train_loader:
            # print(inputs)
            output = model(inputs)
            print(labels_scaler.inverse_transform(output.detach().numpy()))
            if output[0] >= 0.3:
                return "This stock is a strong buy!"
            elif output[0] < 0.3 and output[0] > 0.15:
                return "This stock is a weak buy!"
            elif -0.15 <= output[0] and output[0] <= 0.15:
                return "Hold this stock."
            elif -0.3 < output[0] and -output[0] < -0.15:
                return "This stock is a weak sell!"
            else:
                return "This stock is a strong sell!"
            # print(f"Prediction: {output.item()}")

#Setup; Do not comment
data, labels = loadData()
data, labels, data_scaler, labels_scaler = normalize_data(data, labels)

data = torch.tensor(data, dtype=torch.float32)
labels = torch.tensor(labels, dtype=torch.float32)

dataset = TensorDataset(data, labels)
train_loader = DataLoader(dataset, batch_size=10, shuffle=True)
model = StockModel(4, 256, 2)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

summary(model, (10, 365, 4))

loadModel()

#Comment out what you dont want to run
# trainer(train_loader)
# evaluation(train_loader)
company_list = ["LUMN", "NVDA", "GOOGL", "ZM"] #Sell and buy
for comp in company_list:
    decision = prediction(comp)
    print(decision)

# Discord Bot!

intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)

@client.event
async def on_ready():
    print(f'Bot logged in as {client.user}.')

@client.event
async def on_message(message):
    if message.author == client.user or str(message.author) != "lee98976":
        print(str(message.author))
        return
    
    if message.content.startswith('$Stock '):
        try:
            stockSymbol = message.content.split()[1]
            message1 = prediction(stockSymbol)
            await message.channel.send(message1)
        except:
            print("Invalid Stock")

client.run()
print("Bot logged out.")

'''
To-Do:
1. Organize everything into functions
2. Load from JSON file
3. Improve model
4. Make Discord Bot
2. Load from JSON file #
3. Improve model #
4. Make Discord Bot #
5. Use it on Alpaca
6. Make model save
'''
