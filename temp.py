import os
import random
import time
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from alpaca.data.requests import StockBarsRequest
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.timeframe import TimeFrame
from sklearn.preprocessing import MinMaxScaler

np.set_printoptions(threshold = np.inf)

class StockData(Dataset):
    def __init__(self, stock="NVDA", sample=10, lookback=4, start_date=datetime(2022, 1, 1), end_date=datetime.now()):
        self.stock = stock
        self.sampleSize = sample
        self.lookback = lookback
        self.start_date = start_date
        self.end_date = end_date
        self.data_client = StockHistoricalDataClient('PKZ5HQ89HCEAW6H0QSPI', '1kqpWNWgjTJ6fnEKcJCc5H2lPD719D6iDHE9Ka9L')
        self.fetch_data()
        print(self.data)

    def __len__(self):
        return self.sampleSize

    def __getitem__(self, idx):
        # Extract the sample from the data 
        x = self.data[idx:idx + self.lookback]
        y = self.data[idx + self.lookback, 3]  # Assuming 'Close' is at index 3

        # Normalize the data
        x = MinMaxScaler().transform(x)
        
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def requestData(self, timeframe, start : datetime, end : datetime):
        request_params = StockBarsRequest(
            symbol_or_symbols = [self.stock],
            timeframe = timeframe,
            start=start,
            end=end,
        )
        try:
            bars = self.data_client.get_stock_bars(request_params)
            return bars
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None

    def fetch_data(self):
        # Fetch a large chunk of data
        data = self.requestData(TimeFrame.Hour, self.start_date, self.end_date)
        
        # Process the data (assuming data is not None)
        if data is not None:
            self.data = np.array([[candle.open, candle.high, candle.low, candle.close] for candle in data[self.stock]])
            
            # Fit the scaler to the data
            MinMaxScaler().fit(self.data)
            

        else:
            print("No data fetched, please check your API keys and date range.")

    # def generate_synthetic_data(self):
    #     self.data = self.data[:-self.lookback]
    #     self.data = self.data[np.random.choice(len(self.data), size=self.sampleSize, replace=False)]

# Example usage
dataset = StockData(sample=10, lookback=4)
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)