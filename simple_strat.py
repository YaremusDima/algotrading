import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wget
import torch
import os

'''
#рассчет доходностей
data['r'] = data['Price'][1:].reset_index(drop=True)/data['Price'][:-1].reset_index(drop=True) - 1
r = prices[1:]/prices[:-1] - 1

#рассчет скользящих средник 
short = 5
long = 12
data['SMA'] = data['Price'].rolling(short).mean()
data['LMA'] = data['Price'].rolling(long).mean()

r = prices[1:]/prices[:-1] - 1
short_moving = moving_average(prices, short)[long-short:-1]
long_moving = moving_average(prices, long)[:-1]
r = r[long-1:]
short_moving.size(0), long_moving.size(0), r.size(0)

#Избавляемся от ненужных строк
data = data.drop(len(data)-1)
data = data.drop(data.index[0:long-1])

#рассчитываем доходность стратегии
data['strategy'] = data['r']*(data['SMA']>data['LMA']) - data['r']*(data['SMA']<data['LMA'])

strat = r * (short_moving > long_moving) - r * (short_moving < long_moving)

plt.plot(torch.cumprod(1 + strat, dim=0))
plt.plot(torch.cumprod(1 + r, dim=0))
plt.show()

#визуализируем результаты
plt.plot(np.cumprod(1+data['strategy']))
plt.plot(np.cumprod(1+data['r']))
r_strat = np.prod(1+data['strategy'])**(256/len(data['strategy'])) - 1
r_bnh = np.prod(1+data['r'])**(256/len(data['r'])) - 1
#print('Годовая доходность стратегии скользящих средних '+str(round(r_strat*100,2))+'%')
#print('Годовая доходность стратегии buy and hold '+str(round(r_bnh*100,2))+'%')
'''


# функции для подсчета доходностей
def moving_average(X: torch.Tensor, width: int) -> torch.Tensor:
    ret: torch.Tensor = torch.cumsum(X, dim=0)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width


def profitability_bnh(prices: torch.Tensor):
    """
      Доходность стратегии Buy and hold за весь период
    """
    r = prices[1:] / prices[:-1] - 1
    return torch.prod(1 + r, dim=0) - 1


def profit_bnh_slow(prices: torch.Tensor):
    profit = 1
    for i in range(len(prices)):
        if i != 0:
            profit *= prices[i] / prices[i - 1]
    return profit - 1


def profit_mooving_average(prices, short, long, stoploss, takeprofit):
    x = []
    for i in range(len(prices)):
        pass


def profability_mooving_average(prices: torch.Tensor, short: int, long: int):
    """
      Доходность скользящих средних за весь период
    """
    r = prices[1:] / prices[:-1] - 1
    short_moving = moving_average(prices, short)[long - short:-1]
    long_moving = moving_average(prices, long)[:-1]
    r = r[long - 1:]
    strat = r * (short_moving > long_moving) - r * (short_moving < long_moving)
    return torch.prod(1 + strat, dim=0) - 1


def load_prices(LINK):
    if not os.path.exists('market-price.csv'):
        wget.download(LINK, "market-price.csv")
    data = pd.read_csv('market-price.csv')
    data = data.rename(columns={data.columns[0]: "Date", data.columns[1]: "Price"})[
           :int(len(data.values) * percentage / 100)]
    prices = torch.Tensor(data['Price'].values)
    return data, prices


if __name__ == '__main__':
    # константы
    LINK = 'https://api.blockchain.info/charts/market-price?format=csv'
    LINK2 = ''
    percentage = 100

    # импорт данных
    data, prices = load_prices(LINK)

    #short = 4
    #long = 21
    short = int(input())
    long = int(input())
    assert long > short
    print('Годовая доходность стратегии скользящих средних ' + str(
        round(float(profitability_bnh(prices)) * 100, 2)) + '%')
    print(
        'Годовая доходность стратегии скользящих средних ' + str(round(float(profit_bnh_slow(prices)) * 100, 2)) + '%')
    print('Годовая доходность стратегии buy and hold ' + str(
        round(float(profability_mooving_average(prices, short, long)) * 100, 2)) + '%')
