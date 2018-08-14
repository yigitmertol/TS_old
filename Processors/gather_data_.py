# This script includes the functions that downloads and prepares the data sets
# that are intended to use


## COIN CAP API
import requests
api_key = "81847548-30B2-4BAE-9888-503720BFDE8D"


# def get_hist(market, )


url = 'https://rest.coinapi.io/v1/exchangerate/BTC/USD'
headers = {'X-CoinAPI-Key' : api_key}

url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?' + \
      'period_id=6HRS&'+ \
      'time_start=2016-01-01T00:00:00'+\
      "&time_end={2016-01-02T00:00:00}"

# url = 'https://rest.coinapi.io/v1/ohlcv/BITSTAMP_SPOT_BTC_USD/history?period_id=1MIN&time_start=2016-01-01T00:00:00'

response = requests.get(url, headers=headers)
for info in str(response.content).split(','):
      print(info)


### QUANDL API

import quandl