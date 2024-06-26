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

class StockData(Dataset):
    def __init__(self, sample=10):
        self.sampleSize = sample
        self.data_client = StockHistoricalDataClient('PKZ5HQ89HCEAW6H0QSPI', '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L')
        self.answer, self.inputData = self.returnRandomData()
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
        )
        bars = self.data_client.get_stock_bars(request_params)
        return bars

    def returnRandomData(self):
        data = {"NVDA" : []}

        def helper():
            try:
                if len(data["NVDA"]) != 5:
                    return False
            except:
                return False
            return True

        finalInputs = []
        finalAnswers = []
        for i in range(self.sampleSize):
            #Date
            print(i)
            data = {"NVDA" : []}
            while not helper():
                year = random.randint(2020, 2023)
                month = random.randint(1, 12)
                day = random.randint(1, 28)

                date1 = datetime(year, month, day, 7)
                date2 = datetime(year, month, day, 12)

                data = self.requestData("NVDA", TimeFrame.Hour, date1, date2)

            #Preprocess
            #print(data["NVDA"])

            temp = []
            for candleStick in data["NVDA"]:
                temp.append([candleStick.high, candleStick.low, candleStick.open, candleStick.close])

            answer = temp.pop()[3]
            if answer >= temp[0][3]:
                answer = 0
            else:
                answer = 1

            finalAnswers.append(answer)
            finalInputs.append(temp)

        return finalAnswers, finalInputs