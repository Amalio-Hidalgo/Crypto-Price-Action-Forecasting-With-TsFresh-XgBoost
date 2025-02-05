# CoinGecko API Wrapper Module:
# This module provides wrapper functions for accessing CoinGecko cryptocurrency data. 
# Main Functions (1&2) use others to gather price data into melted long format Pandas or Dask DataFrames.
# Outputs are in the right format for data rolling and feature extraction with TsFresh.

# Dependencies:
#     - pandas
#     - requests
#     - datetime
#     - dask (for distributed computing)
import requests, time 
import pandas as pd
import datetime as dt
import dask.dataframe as dd

# 1. CoinGecko Historical Price Data- Pandas  
# Creates a melted (long format) Pandas DataFrame sharing a resampled index converted to your timezone.
# Parameters:
#     timeframe (int): Historical data period in days
#     top_coins (int): Number of top coins by market cap to include
#     periods (int): Resampling period:
        # Adjust periods to increase or decrease granularity without losing regularity of time intervals/ homogenous timestamp spacing. 
        # If timeframe = 1, CoinGecko defaults to 5min granularity, else 1hr.
        # (eg:timeframe =1, periods=3->15min gran., timeframe=50, periods=4->4hr gran.)
#     api_key (str): CoinGecko API key
#     timezone: timezone in pandas recognizable format (e.g. 'CET', 'UTC', 'CET1', 'UTC5')
# Returns:
#     pandas.DataFrame: Melted long format DataFrame with datetime index, containing price, market cap, and volume data
def CoinGecko_HSPD_Pandas(timeframe, top_coins, periods, api_key, timezone='CET'):
    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": api_key
    }
    coins = CoinGecko_TopCoinsMC_Pandas(number= top_coins, headers=headers)
    count=0
    for coin in coins:
        response = pd.DataFrame(Coingecko_HSPD_Json(coin=coin, timeframe=timeframe, headers=headers))
        response['datetime'] = response['prices'].map(lambda x: x[0]).apply(pd.to_datetime, unit='ms', utc=True)
        for column in response.columns.drop('datetime'):
            response[column]= response[column].map(lambda x: x[1])
        if timeframe<=1: 
            freq= f'{periods*5}min'
            response = response.set_index('datetime', drop=True).resample(freq).last()
            response.index= response.index.tz_convert(timezone)
        else: 
            freq= f'{periods}h'
            response = response.set_index('datetime', drop=True).resample(freq).last()
            response.index= response.index.tz_convert(timezone)
        if count == 0: 
            output = response
        else: 
            output= output.join(response, rsuffix=' '+ coin, how='left')
        count=count+1
    return output.melt(ignore_index=False).reset_index(drop=False)

# 2. CoinGecko-Dask: Historical Price Data for All Coins Collected Into Flat Dask DataFrame Sharing Resampled Index 
# Creates Dask Distributed DataFrame with/without resampled historical price data using previous functions
# Parameters:
#     timeframe (int): Historical data period in days
#     top_coins (int): Number of top coins by market cap to include
#     periods (int): Resampling period:
        # Adjust periods to increase or decrease granularity without losing regularity of time intervals/ homogenous timestamp spacing. 
        # If timeframe = 1, CoinGecko defaults to 5min granularity, else 1hr.
        # (eg:timeframe =1, periods=3->15min gran., timeframe=50, periods=4->4hr gran.)
#     api_key (str): CoinGecko API key
#     timezone: timezone in pandas recognizable format (e.g. 'CET', 'UTC', 'CET1', 'UTC5')
# Returns:
#     pandas.DataFrame: Melted long format DataFrame with datetime index, containing price, market cap, and volume data
def CoinGecko_HSPD_Dask(timeframe, top_coins, periods, api_key, timezone='CET', splits=5):
    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": api_key
    }
    coins = CoinGecko_TopCoinsMC_Pandas(top_coins, headers=headers)
    count=0
    for coin in coins:
        dd_response= Coingecko_HSPD_Json(coin, headers=headers, timeframe=timeframe)
        dd_response= dd.from_dict(dd_response, npartitions=1)
        dd_response['datetime'] = dd_response['prices'].map(lambda x: float(x.split('[')[1].split(',')[0]), meta=('datetime', 'float64'))
        func = lambda x : pd.to_datetime(x, unit='ms', origin='unix', utc=True)
        dd_response['datetime']= dd_response['datetime'].map(func, meta=('datetime', 'datetime64[ns, UTC]'))
        if timeframe<=1: freq= f'{periods*5}min'
        else: freq= f'{periods}h'
        dd_response= dd_response.set_index('datetime', sorted=False, npartitions=len(dd_response.columns), drop=True).resample(freq).last()
        func= lambda x : float(x.split('[')[1].split(',')[1].split(']')[0])
        for column in dd_response.columns:
            dd_response[column]= dd_response[column].compute().dropna().apply(func)
        if count == 0 : output=dd_response
        else: output= output.join(dd_response, how='left', rsuffix=' '+coin)
        count= count+1
    output= output.reset_index().melt('datetime').sort_values(['variable','datetime'])
    output['datetime']= output['datetime'].dt.tz_convert(timezone)
    output = output.repartition(npartitions= splits)
    return output


# Supporting Functions//
# CoinGecko Historical Price Data- JSON
# Fetches historical price data from CoinGecko API in JSON format.
# Parameters:
#     coin (str): Cryptocurrency CoinGecko ID 
#     (e.g.'bitcoin', 'ethereum', 'ripple', 'tether', 'solana')
#     headers (dict): API request headers including authentication 
#     (e.g. {"accept": "application/json", "x-cg-demo-api-key": api_key})
#     timeframe (int): Historical data period in days (default: 1) 
# **Important Note: CoinGecko only offers 5min granularity for 1 Day timeframe and no hourly data past timeframe= 89
# Returns:
#     dict: JSON response containing price, market cap, and volume data
def Coingecko_HSPD_Json(coin, headers, timeframe):
    start= int(dt.datetime.now().timestamp())
    end= int((dt.datetime.now() - dt.timedelta(days=timeframe)).timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/"+ coin + f"/market_chart/range?vs_currency=usd&from={end}&to={start}"
    return  requests.get(url, headers=headers).json()
# CoinGecko:ID Map- Pandas
# Fetches historical price data from CoinGecko API in JSON format.
# Parameters:
#     coin (str): Cryptocurrency identifier (e.g., 'bitcoin')
#     headers (dict): API request headers including authentication
#     timeframe (int): Historical data period in days (default: 1)
# Returns:
#     dict: JSON response containing price, market cap, and volume data
def CoinGecko_IDs_Pandas(coins, headers):
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url, headers=headers)
    id_map = pd.DataFrame(response.json()).set_index('name')
    return id_map.loc[coins]
# CoinGecko: Top Coins by MarketCap- Pandas
# Fetches CoinGecko ids of top n coins from  API in JSON format.
# Parameters:
#     number (int): Cryptocurrency identifier (e.g., 'bitcoin')
#     headers (dict): API request headers including authentication
# Returns:
#     Pandas df of response containing ids of top n coins by MarketCap
def CoinGecko_TopCoinsMC_Pandas(number, headers):
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    response = pd.DataFrame(requests.get(url, headers=headers).json())
    investment_universe = response.head(number)['id'].values
    return investment_universe
