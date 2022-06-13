import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wget
import torch
import os
"""
    Buy and hold strategy
    Moving average strategy
    Brute force
"""

# функции для подсчета доходностей
def moving_average(X, width: int):
    ret = np.cumsum(X, axis=0)
    ret[width:] = ret[width:] - ret[:-width]
    return ret[width - 1:] / width


def profitability_bnh(prices):
    """
      Доходность стратегии Buy and hold за весь период
    """
    r = prices[1:] / prices[:-1] - 1
    return np.prod(1 + r, axis=0) - 1


def profit_bnh_slow(prices):
    """
        Доходность стратегии Buy and hold за весь период
        Slow function
    """
    profit = 1
    for i in range(len(prices)):
        if i != 0:
            profit *= prices[i] / prices[i - 1]
    return profit - 1


def profit_moving_average(prices, short, long, stoploss, takeprofit):
    """
        stoploss: value in range (0, 1)
        takeprofit: value in range (0, 1)
    """
    wallet = 0
    r = prices[1:] / prices[:-1] - 1
    short_moving = moving_average(prices, short)[long - short:-1]
    long_moving = moving_average(prices, long)[:-1]
    assert len(short_moving) == len(long_moving)
    r = r[long - 1:]
    p = prices[long:]
    sdel = {'sell': [], 'buy': []}
    for i in range(len(short_moving)):
        if i == 0:
            continue
        # открытие сделок
        if short_moving[i - 1] < long_moving[i - 1] and short_moving[i] > long_moving[i]:
            sdel['buy'].append(p[i])
        elif short_moving[i - 1] > long_moving[i - 1] and short_moving[i] < long_moving[i]:
            sdel['sell'].append(p[i])
        # закрытие сделок
        for price in sdel['buy']:
            if p[i] / price - 1 >= takeprofit or p[i] / price - 1 <= -stoploss:
                wallet += p[i] - price
                sdel['buy'].pop(sdel['buy'].index(price))
        for price in sdel['sell']:
            if price / p[i] - 1 >= takeprofit or price / p[i] - 1 <= -stoploss:
                wallet += price - p[i]
                sdel['sell'].pop(sdel['sell'].index(price))
    return wallet / p[1]


def profability_moving_average_slow(prices, short, long):
    wallet = 1
    r = prices[1:] / prices[:-1] - 1
    short_moving = moving_average(prices, short)[long - short:-1]
    long_moving = moving_average(prices, long)[:-1]
    r = r[long - 1:]
    for r_, s, l in zip(r, short_moving, long_moving):
        if (s > l):
            wallet *= 1 + r_
        else:
            wallet *= 1 - r_
    return wallet - 1


def profability_moving_average(prices, short: int, long: int):
    """
      Доходность скользящих средних за весь период
    """
    r = prices[1:] / prices[:-1] - 1
    short_moving = moving_average(prices, short)[long - short:-1]
    long_moving = moving_average(prices, long)[:-1]
    r = r[long - 1:]
    strat = r * (short_moving > long_moving) - r * (short_moving < long_moving)
    return np.prod(1 + strat, axis=0) - 1


def load_prices(LINK, percentage):
    if not os.path.exists('data/market-price.csv'):
        wget.download(LINK, "data/market-price.csv")
    data = pd.read_csv('data/market-price.csv')
    data = data.rename(columns={data.columns[0]: "Date", data.columns[1]: "Price"})[
           :int(len(data.values) * percentage / 100)]
    prices = np.array(data['Price'].values)
    return data, prices


if __name__ == '__main__':
    # константы
    LINK = 'https://api.blockchain.info/charts/market-price?format=csv'


    # импорт данных
    data, prices = load_prices(LINK, 100)

    short = 4
    long = 21
    stoploss = 0.002
    takeprofit = 0.006
    print('Годовая доходность стратеги buy and hold ' + str(
        round(float(profitability_bnh(prices)) * 100, 2)) + '%')
    print(
        'Годовая доходность стратегии buy and hold(медленная) ' + str(
            round(float(profit_bnh_slow(prices)) * 100, 2)) + '%')
    print('Годовая доходность стратегии скользящих средних (2 параметра)' + str(
        round(float(profability_moving_average(prices, short, long)) * 100, 2)) + '%')
    print('Годовая доходность стратегии скользящих средних (2 параметра, медленная) ' + str(
        round(float(profability_moving_average_slow(prices, short, long)) * 100, 2)) + '%')
    print("prices[-1] / prices[0] - 1: ", prices[-1] / prices[0] - 1)
    print(profit_moving_average(prices, short, long, stoploss, takeprofit))
