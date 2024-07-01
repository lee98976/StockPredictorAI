import os
import random
import time
from datetime import datetime

import numpy
import torch
from torch.utils.data import Dataset, DataLoader
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from alpaca.data.enums import Adjustment

class StockData(Dataset):
    def __init__(self, stockSymbols, sample=10):
        self.sampleSize = sample
        self.stockSymbols = stockSymbols
        self.data_client = StockHistoricalDataClient('PKZ5HQ89HCEAW6H0QSPI', '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L')
        self.answer, self.inputData = self.returnRandomData()
        # print(numpy.ndarray(3, self.inputData).size)

            
    def __len__(self):
        return self.sampleSize
    
    def __getitem__(self, idx):
        return self.inputData[idx], self.answer[idx]

    def requestData(self, stock : list, timeframe, start : datetime, end : datetime):
        request_params = StockBarsRequest(
            symbol_or_symbols = stock,
            timeframe = timeframe,
            start=start,
            end=end,
            adjustment=Adjustment('all')
        )

        bars = self.data_client.get_stock_bars(request_params)
        return bars

    def returnRandomData(self):
        allData = self.requestData(self.stockSymbols, TimeFrame.Day, datetime(2020, 1, 1), datetime(datetime.now().year, datetime.now().month, datetime.now().day))
        # data = data["NVDA"]

        finalAnswers, finalInputs = [], []

        for i in range(self.sampleSize):
            data = allData[random.choice(list(allData.data.keys()))] #Select a random stock to take the 365 days from
            
            rangeStart = random.randint(1, len(data) - 400)
            dataRange = data[rangeStart:rangeStart + 380]

            temp = []

            for candleStick in dataRange:
                temp.append([candleStick.high, candleStick.low, candleStick.open, candleStick.close])

            answer = temp.pop()[3]  # Use the closing price of the last day as the label for regression
            temp = temp[0:365]

            finalAnswers.append(answer)
            finalInputs.append(temp)

        return finalAnswers, finalInputs

class RealTimeData(Dataset):
    def __init__(self, stockSymbol):
        self.stockSymbol = stockSymbol
        self.data_client = StockHistoricalDataClient('PKZ5HQ89HCEAW6H0QSPI', '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L')
        self.answer, self.inputData = self.returnRandomData()

    def requestData(self, stock : str, timeframe, start : datetime, end : datetime):
        request_params = StockBarsRequest(
            symbol_or_symbols = [stock],
            timeframe = timeframe,
            start=start,
            end=end,
            adjustment=Adjustment('all')
        )
        bars = self.data_client.get_stock_bars(request_params)
        return bars

    def returnRandomData(self):
        data = self.requestData(self.stockSymbol, TimeFrame.Day, datetime(2020, 1, 1), datetime(datetime.now().year, datetime.now().month, datetime.now().day))
        data = data[self.stockSymbol]

        finalAnswers, temp = [], []

        dataRange = data[-365:]

        for candleStick in dataRange:
            temp.append([candleStick.high, candleStick.low, candleStick.open, candleStick.close])

        finalAnswers.append([0])

        return finalAnswers, temp
