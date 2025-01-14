# Ethereum_Price_Action
This is project is aimed at deriving a forecast of Ethereum's price action (log-returns and/or volatility). 
The general framework consists of:
1. API calls to one or various data providers using the functions available in My_API_Wraps- Currently supports Coingecko, Lunarcrush, Token-Terminal, and Dune. Each of my API wraps return the data in the melted format neccesary for the next step:
2. Data Rolling & Feature Extraction with TsFresh using functions available in My_FE_Wraps https://tsfresh.readthedocs.io/en/latest/text/introduction.html.
3. (Optional) Adding Additional Features: functions available in My_FE_Wraps.
4. Feature Selection with TsFresh with functions available in My_FS_Wraps.  
5. Preliminary XgBoost model & Hypertuning of Model with RandomGridSearch or Optuna with func. ava. in My_ML_Wraps
6. Final Model- Visualization & Recording of Results 
7. Pancakeswap Trading Bot
 
Multiple paths have been tested and are available:
1. With Pandas and Multiprocessing
2. With Pandas and Dask Distributor
3. With Dask on the CPU
4. With Dask on the GPU
