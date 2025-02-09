# Crypto Price Action Forecasting With TsFresh & XgBoost
This is project was conceived with the intention of deriving a forecast of Ethereum's price action (log-returns and/or volatility), but can be used to predict the price action of any cryptocurrency subject to the condition that the assets historical data can be accessed via the data providers' APIs that my code currently supports.  

## The general framework consists of:
1. API calls to one or various data providers using the functions available in My_API_Wraps- Currently supports Coingecko, In the process of updating with Lunarcrush, Token-Terminal, and Dune. Each of my API wraps return the data in the melted format neccesary for the next step;
2. Data Rolling & Feature Extraction with TsFresh using functions available in My_FE_Wraps https://tsfresh.readthedocs.io/en/latest/text/introduction.html. My feature extraction functions return the data in a pivoted format- relative to the API wraps output- that is neccesary for the feature selection process in the next step;
 2a. WIP- (Optional) Adding Additional Features (macroeconomic, tokenomics, social metrics...): in the process of updating functions available in My_FE_Wraps.
4. Feature Selection with TsFresh with functions available in My_FS_Wraps.  
5. Preliminary XgBoost model & Hyperparameter tuning of Model with RandomGridSearch or Optuna (Example in crypto price action notebook).
6. Final Model- Visualization & Recording of Results 
7. Pancakeswap Trading Bot-> In progress...

## Multiple paths have been tested and are available:
1. With Pandas and Multiprocessing
2. With Pandas and Dask Distributor for Feature Extraction
3. With Dask end-to-end
   
Pandas and Multiprocessing is by far the least complicated. However, due to the extreme number of features extracted by tsfresh standing at 4662 per variable, if the user wishes to use more than a couple cryptocurrencies over more than a daily timeframe they will most likely run into memory constraints. As such, two alternative methods are available. Pandas and the Dask Distributor keeps data locally on pandas but uses a dask distributor in the extraction of features connected to a client and cluster specified by the user via a scheduler address parameter. The other uses Dask to distribute the computations and persist intermediate results from start to finish and as a result is markedly more complicated but handles large datasets far better, even those that don't fit in memory.

