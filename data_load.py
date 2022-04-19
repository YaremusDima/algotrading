import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import wget
import torch
import os
import requests

from datetime import datetime, timedelta

from tinkoff.invest import CandleInterval, Client


def load_prices(LINK, persentage = 100):
    if not os.path.exists('market-price.csv'):
        wget.download(LINK, "market-price.csv")
    data = pd.read_csv('market-price.csv')
    data = data.rename(columns={data.columns[0]: "Date", data.columns[1]: "Price"})[
           :int(len(data.values) * persentage / 100)]
    prices = torch.Tensor(data['Price'].values)
    return data, prices


def load_dollar(date1, date2=None):
    """
        date1: "dd/mm/yyyy"
    """
    api_key = 'MTIJZ4YNF5M7NIO5'
    if date2 is None:
        date2 = datetime.now()
    date1 = datetime.strptime(date1, "%d/%m/%Y")
    get_xml = requests.get(
        f'http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1={date1.day}/{date1.month}/{date1.year}&date_req2={date2.day}/{date2.month}/{date2.year}&VAL_NM_RQ=R01235'
    )
    df = pd.read_xml(get_xml)
    df.head()


def main():
    # константы
    LINK = 'https://api.blockchain.info/charts/market-price?format=csv'
    TOKEN = 't.UNbUu1mBSe7f6bY1ryEAQtnUzpEwzDzTHens3ROw8jg-4lJ4MNkQRktC2TKI1kS6szlaCUPZCyDQbTHbb67G4w'

    # импорт данных
    data, prices = load_prices(LINK)

    with Client(TOKEN) as client:
        stocks = client.market.market_stocks_get()
        print(stocks)


if __name__ == '__main__':
    main()
