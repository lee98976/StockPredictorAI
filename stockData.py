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
    def __init__(self, isRealTime, sample=10):
        self.sampleSize = sample
        self.data_client = StockHistoricalDataClient('PKZ5HQ89HCEAW6H0QSPI', '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L')
        self.isRealTime = isRealTime
        self.answer, self.inputData = self.returnRandomData()
        # print(numpy.ndarray(3, self.inputData).size)

        #NORMALIZE LATER
            
    def __len__(self):
        return self.sampleSize
    
    def __getitem__(self, idx):
        return self.inputData[idx], self.answer[idx]

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

    # def fit_transform(self, x): #Normalize data to -1 to 1
    #     mu = numpy.mean(x, axis=(0), keepdims=True)
    #     sd = numpy.std(x, axis=(0), keepdims=True)
    #     normalized_x = (x - mu)/sd
    #     return normalized_x

    def returnRandomData(self):
        data = self.requestData("NVDA", TimeFrame.Day, datetime(2020, 1, 1), datetime(datetime.now().year, datetime.now().month, datetime.now().day))
        data = data["NVDA"]
        print(len(data), type(data))
        finalAnswers, finalInputs = [], []
        
        if not self.isRealTime: amount = self.sampleSize
        else: amount = 1

        for i in range(amount):
            if not self.isRealTime: 
                rangeStart = random.randint(1, 600)
                dataRange = data[rangeStart:rangeStart+365]
            else:
                dataRange = data[-365:]
            temp = []

            for candleStick in dataRange:
                temp.append([candleStick.high, candleStick.low, candleStick.open, candleStick.close])

            if not self.isRealTime:
                answer = temp.pop()[3]
                #Answer = [buy, sell]
                if answer >= temp[-1][3]:
                    answer = [1, 0]
                else:
                    answer = [0, 1]
            else:
                answer = None

            # answer = numpy.array(answer)
            # answer = self.fit_transform(answer).tolist()

            finalAnswers.append(answer)
            finalInputs.append(temp)

        return finalAnswers, finalInputs