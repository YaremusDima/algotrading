import numpy as np
import pandas as pd
import wget
import torch
import os
import random
from datetime import datetime


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


def profit_moving_average(prices, short, long, stoploss, takeprofit):
    """
        stoploss: value in range (0, 1)
        takeprofit: value in range (0, 1)
    """
    assert long > short
    wallet = 0
    r = prices[1:] / prices[:-1] - 1
    short_moving = moving_average(prices, short)[long - short:]
    long_moving = moving_average(prices, long)
    assert len(short_moving) == len(long_moving)
    r = r[long - 2:]
    p = prices[long - 1:]
    flag = 0
    flags = torch.zeros_like(short_moving)
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
    #if not len(sdel['buy']) == 0:
    #    print("buy: ", sdel['buy'])
    #if not len(sdel['sell']) == 0:
    #    print("sell: ", sdel['sell'])
    #print(sdel)
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



def profability_moving_average(prices: torch.Tensor, short: int, long: int):
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

def sgd_step(x_0:int, y_0:int, z_0:float, w_0:float):
    n = 0.0005
    k = 1
    a = profit_moving_average(prices, x_0,y_0,z_0,w_0)
    if y_0 > x_0+k:
        b = profit_moving_average(prices, x_0 + k,y_0,z_0,w_0)
    else:
        b = -1000000
    c = profit_moving_average(prices, x_0 - k,y_0,z_0,w_0)
    d = profit_moving_average(prices, x_0,y_0 + k,z_0,w_0)
    if y_0 - k > x_0:
        e = profit_moving_average(prices, x_0,y_0 - k,z_0,w_0)
    else:
        e = -1000000
    f = profit_moving_average(prices, x_0,y_0, z_0 + n, w_0)
    if z_0 - n > 0:
        g = profit_moving_average(prices, x_0,y_0, z_0 - n, w_0)
    else:
        g = -1000000
    h = profit_moving_average(prices, x_0,y_0, z_0, w_0 + n)
    if w_0 - n > 0:
        i = profit_moving_average(prices, x_0,y_0, z_0, w_0 - n)
    else:
        i = -1000000
    if a == max(a,b,c,d,e,f,g,h,i):
        return x_0,y_0,z_0,w_0
    elif b == max(a,b,c,d,e,f,g,h,i):
        return (x_0 + k,y_0,z_0,w_0)
    elif c == max(a,b,c,d,e,f,g,h,i):
        return (x_0 - k,y_0,z_0,w_0)
    elif d == max(a,b,c,d,e,f,g,h,i):
        return (x_0,y_0 + k,z_0,w_0)
    elif e == max(a, b, c, d, e, f, g, h, i):
        return (x_0, y_0 - k, z_0, w_0)
    elif f == max(a, b, c, d, e, f, g, h, i):
        return (x_0, y_0, z_0 + n, w_0)
    elif g == max(a, b, c, d, e, f, g, h, i):
            return (x_0, y_0, z_0 - n, w_0)
    elif h == max(a, b, c, d, e, f, g, h, i):
        return (x_0, y_0, z_0, w_0 + n)
    elif i == max(a, b, c, d, e, f, g, h, i):
            return (x_0, y_0, z_0, w_0 - n)



def sgd(x_0:int, y_0:int, z_0:float, w_0:float):
    (x,y,z,w) = sgd_step(x_0,y_0,z_0,w_0)
    while (x,y,z,w) != (x_0,y_0,z_0,w_0):
        x_0,y_0,z_0,w_0 = x,y,z,w
        (x,y,z,w) = sgd_step(x_0,y_0,z_0,w_0)
    return x_0,y_0,z_0,w_0

def sgd_for_all(n:int):
  l=[]
  x=np.zeros(n, dtype=int)
  y=np.zeros(n, dtype=int)
  x_0=np.zeros(n, dtype=int)
  y_0=np.zeros(n, dtype=int)
  z=np.zeros(n, dtype=float)
  w=np.zeros(n, dtype=float)
  z_0 = np.zeros(n, dtype=float)
  w_0 = np.zeros(n, dtype=float)
  for i in range(n):
    x[i] = random.randint(2,20)
    y[i] = random.randint(x[i] + 1,50)
    z[i] = random.uniform(0.0001,0.5)
    w[i] = random.uniform(0.0001,0.9)
    #print(x[i], y[i])
    x_0[i],y_0[i],z_0[i],w_0[i] = sgd(x[i],y[i],z[i],w[i])
    #print(int(x_0[i]), int(y_0[i]))
    l.append(float(profit_moving_average(prices, int(x_0[i]), int(y_0[i]), z_0[i],w_0[i])))
  maximize = max(l)
  idx = l.index(maximize)
  s = (x_0[idx], y_0[idx], z_0[idx], w_0[idx])

  return maximize, s




if __name__ == '__main__':
    # константы
    LINK = 'https://api.blockchain.info/charts/market-price?format=csv'
    LINK2 = ''
    percentage = 100

    # импорт данных
    data, prices = load_prices(LINK)

    short = 4
    long = 21
    stoploss = 0.002
    takeprofit = 0.006
    # short = int(input())
    # long = int(input())
    assert long > short
    #print('Годовая доходность стратеги buy and hold ' + str(
    #    round(float(profitability_bnh(prices)) * 100, 2)) + '%')
    #print(
    #    'Годовая доходность стратегии buy and hold(медленная) ' + str(
    #        round(float(profit_bnh_slow(prices)) * 100, 2)) + '%')
    #print('Годовая доходность стратегии скользящих средних (2 параметра)' + str(
    #    round(float(profability_moving_average(prices, short, long)) * 100, 2)) + '%')
    #print('Годовая доходность стратегии скользящих средних (2 параметра, медленная) ' + str(
    #    round(float(profability_moving_average_slow(prices, short, long)) * 100, 2)) + '%')
    #print("prices[-1] / prices[0] - 1: ", prices[-1] / prices[0] - 1)
    #print(profit_moving_average(prices, short, long, stoploss, takeprofit))

    start = datetime.now()
    res, params = sgd_for_all(10)
    print("Максимальная выручка: ", res, "Параметры: ", params)
    print("время исполнения: ", datetime.now() - start)
