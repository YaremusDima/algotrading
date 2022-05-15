from openapi_client import openapi

TOKEN = 't.dgU9Chb1QhHvp2M5OooBF7HF8X7-5KGTnPQbgta6OOi2oJ6OC-BjguDz6sL4u13cNpHVY-8HHs0FMI4MjZEthA'
SANDBOX_TOKEN = 't.Syh0-f9FlbMG9iQHXIjI9tjLjl24qNJleHITjUHj_vwNf8OGbGFINKM_Aq08HO2Ucdp6ibS_wcDrSpym-a57bA'
FULL_TOKEN = ''


class MyClient:
    def __init__(self, token=SANDBOX_TOKEN):
        self.token = token
        self.client = openapi.api_client(self.token)
        self.pf = self.client.portfolio.portfolio_get()

    def get_my_stocks(self):
        stock_info = {}
        for stock in self.pf.payload.positions:
            stock_info[stock.name] = {
                "value": stock.average_position_price.value,
                "currency": stock.average_position_price.currency,
                "balance": stock.balance,
                "figi": stock.figi,
                "ticker": stock.ticker,
            }
        return stock_info

    def buy_stock(self, figi, lots):
        info = self.client.orders.orders_market_order_post(figi=figi,
                                                           market_order_request={
                                                               "lots": lots,
                                                               "operation": "Buy"
                                                           }
                                                           )
        print(f"Succesfully made market order to buy stock with figi = {figi}")
        return info

    def sell_stock(self, figi, lots):
        info = self.client.orders.orders_market_order_post(figi=figi,
                                                           market_order_request={
                                                               "lots": lots,
                                                               "operation": "Sell"
                                                           }
                                                           )
        print(f"Succesfully made market order to sell stock with figi = {figi}")
        return info
