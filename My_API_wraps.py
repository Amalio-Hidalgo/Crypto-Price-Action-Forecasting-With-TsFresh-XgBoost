
# API REQUESTS AND JSON READ FUNCTIONS
# -Inputs = Coins, Timeframe

import requests, time 
import pandas as pd
import datetime as dt
import dask.dataframe as dd


# CoinGecko:ID Map
def CoinGecko_IDs_Pandas(coins, headers):
    url = "https://api.coingecko.com/api/v3/coins/list"
    response = requests.get(url, headers=headers)
    id_map = pd.DataFrame(response.json()).set_index('name')
    return id_map.loc[coins]

# CoinGecko: Top Coins by Marketcap- Investment Universe
def CoinGecko_TopCoinsMC_Pandas(number, headers):
    url = "https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd"
    response = pd.DataFrame(requests.get(url, headers=headers).json())
    investment_universe = response.head(number)['id'].values
    return investment_universe

# CoinGecko_Json: Json of CoinGecko Historical Price Data Request
def Coingecko_HSPD_Json(coin, headers, timeframe=1):
    start= int(dt.datetime.now().timestamp())
    end= int((dt.datetime.now() - dt.timedelta(days=timeframe)).timestamp())
    url = f"https://api.coingecko.com/api/v3/coins/"+ coin + f"/market_chart/range?vs_currency=usd&from={end}&to={start}"
    return  requests.get(url, headers=headers).json()

# CoinGecko-Pandas: Historical Price Data for All Coins Collected Into Flat Pandas DataFrame Sharing Resampled Index 
def CoinGecko_HSPD_Pandas(timeframe, top_coins, periods, api_key):
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
            response.index= response.index.tz_convert('CET')
        else: 
            freq= f'{periods}h'
            response = response.set_index('datetime', drop=True).resample(freq).last()
            response.index= response.index.tz_convert('CET')
        if count == 0: 
            output = response
        else: 
            output= output.join(response, rsuffix=': '+ coin, how='left')
        count=count+1
    return output.melt(ignore_index=False).reset_index(drop=False)

# CoinGecko-Dask: Historical Price Data for All Coins Collected Into Flat Dask DataFrame Sharing Resampled Index 
def CoinGecko_HSPD_Dask_CPU(timeframe, top_coins, periods, api_key):
    headers = {
    "accept": "application/json",
    "x-cg-demo-api-key": api_key
    }
    coins = CoinGecko_TopCoinsMC_Pandas(top_coins, headers=headers)
    count=0
    for coin in coins:
        dd_response= Coingecko_HSPD_Json(coin, headers)
        dd_response= dd.from_dict(dd_response, npartitions=1)
        dd_response['datetime'] = dd_response['prices'].map(lambda x: float(x.split('[')[1].split(',')[0]), meta=('datetime', 'float64'))
        func = lambda x : pd.to_datetime(x, unit='ms', origin='unix')
        dd_response['datetime']= dd_response['datetime'].map(func, meta=('datetime', 'datetime64[ns, CET]'))
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
    npart = output['variable'].nunique().compute()
    output= output.repartition(npartitions= int(npart)+1)
    return output


