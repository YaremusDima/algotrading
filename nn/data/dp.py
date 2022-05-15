import numpy as np
import torch
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import pandas_datareader as web
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd


def get_data(ticker, start, end=None):
    if end is None:
        data = web.DataReader(ticker, "yahoo", start, datetime.now())
    else:
        data = web.DataReader(ticker, "yahoo", start, end)
    return data[["Close"]]


class PricesDataset(Dataset):
    """"""

    def __init__(self, data, lookback):
        assert len(data) > lookback
        self.data = data.values.reshape(-1, 1)
        self.lookback = lookback
        #self.scaler = MinMaxScaler()
        #data_scaled = self.scaler.fit_transform(self.data)
        data_scaled = self.data

        data = []
        for index in range(len(data_scaled) - lookback):
            data.append(data_scaled[index: index + lookback])

        data = np.array(data)
        self.x = data[:, :-1, :]
        self.y = data[:, -1, :]
        assert len(self.x) == len(self.y)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]).type(torch.Tensor).reshape(-1, self.lookback - 1), \
               torch.from_numpy(self.y[idx]).type(torch.Tensor)


def get_dataloaders(ticker, lookback, start, batch_size=1, end=None, test_ratio=0.2):
    data = get_data(ticker, start, end)
    test_size = int(np.round(test_ratio * data.shape[0]))
    train_size = data.shape[0] - test_size

    train = data[:train_size]
    test = data[train_size:]

    train_ds = PricesDataset(train, lookback)
    test_ds = PricesDataset(test, lookback)
    train_dl = DataLoader(train_ds, shuffle=False, batch_size=batch_size)
    test_dl = DataLoader(test_ds, shuffle=False, batch_size=batch_size)
    return train_dl, test_dl

