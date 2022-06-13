#from TinkoffAPI.client import MyClient, SANDBOX_TOKEN  # , FULL_TOKEN
from data import dp
from model import model
from datetime import datetime
import torch
from torch.optim import Adam
import numpy as np

'''
client = MyClient(SANDBOX_TOKEN)
for stock, info in client.get_my_stocks().items():
    print(stock)
    print(f"figi={info['figi']}")
'''

start = datetime(2019, 1, 1)

train_dl, test_dl = dp.get_dataloaders("SBER.ME", 20, start, batch_size=2)
for x, y in train_dl:
    print(x.shape, y.shape)
    print(x[0], y[0])
    break