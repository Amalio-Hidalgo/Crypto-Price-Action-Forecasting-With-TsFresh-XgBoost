
# Feature Selection Wrapper Module: Provides feature selection functionality for time series data using TsFresh.
    # Works with both Dask and Pandas dfs.
    # Input df must be in wide format; columns are variables, rows are datetime index. 
    # This is the same format as the output of my FE_wraps.py functions.
    # Adds transformations of target : 
    #   1. Percentage log returns (LR(%)), 
    #   2. 1 period realised % volatility, (1PeriodVol(%))
    # The target variable is the variable to be forecasted:
    #  can be forecasted without any transformation of the variable
        # specify variable by setting target = to a string of the following fom 'variable target'
    #       (eg.: target= 'prices ethereum', target = 'marketcaps bitcoin',...)
    #  can also target transformation (log returns or volatility) of a variable by setting either of Booleans (LR and Vol) to True

# Dependencies:
#     - tsfresh
#     - pandas
#     - numpy
#     - dask
from tsfresh.utilities.dataframe_functions import impute
from tsfresh import select_features
import datetime as dt
import pandas as pd
import numpy as np
import dask.dataframe as dd


# 1. Feature selection on Pandas DataFrame using tsfresh.
    # Inputs:
    #     features (pandas.DataFrame): Wide format DataFrame with datetime index
    #     timeframe (int): Historical data period in days
    #     forecast_periods (int): Number of periods to forecast
    #     p_value (float): Significance level for feature selection
    #     target (str): Target variable for prediction
    # Returns:
    #     pandas.DataFrame: Selected features with target variables
    # Notes:
    #     - Uses local multiprocessing
def SF_Pandas_v1(features, timeframe=1, forecast_periods=1, p_value=0.05, target='prices ethereum', LR=True, Vol=False):
    # Feature selection on Pandas dataframe with Tsfresh
    target_var = features[target]
    features['LR(%)']= np.log(target_var)- np.log(target_var.shift(1))
    features['LR(%)']=features['LR(%)']*100
    features['1PeriodVol(%)']= abs(features['LR(%)'])
    if LR== True: target_var= features['LR(%)']
    elif Vol== True: target_var= features['1PeriodVol(%)']
    elif (LR==False & Vol==False): target_var= target_var
    features['y_future']= target_var.shift(-forecast_periods)
    features[['LR(%)','y_future','prices ethereum','1PeriodVol(%)']].tail(30).plot(subplots=True,figsize= (25, 25))
    FC= features.dropna(subset='y_future')
    FC = impute(FC)
    X= FC.drop('y_future', axis=1)
    y= FC['y_future']
    selected_features= select_features(X, y, p_value, hypotheses_independent=False, n_jobs=14).columns.tolist()
    selected_features.append(target)
    selected_features.append('y_future')
    if target_var.columns[0] != target: selected_features.append(target)
    if target_var.columns[0] != 'LR(%)': selected_features.append('LR(%)')
    if target_var.columns[0] != '1PeriodVol(%)':selected_features.append('1PeriodVol(%)')
    if len (selected_features) <= 5: print('not_enough_sig_feat')
    else: print('enough_sig_feat')
    final_features = FC[selected_features]
    return FC

# 2. Distributed feature selection using Dask and tsfresh.
    # Inputs:
    #     features (dask.DataFrame): Wide format Dask DataFrame
    #     timeframe (int): Historical data period in days
    #     forecast_periods (int): Number of periods to forecast
    #     p_value (float): Significance level for feature selection
    #     target (str): Target variable for prediction
    # Returns:
    #     dask.DataFrame: Selected features with target variables
    # Notes:
    #     - Optimized for distributed processing
    #     - Persists intermediate results
    #     - Partition-wise computation
def SF_Dask_v1(features, timeframe=1, forecast_periods=1, p_value=0.05, target='prices ethereum',LR=True, Vol=False):
    features['LR(%)']= np.log(features[target])- np.log(features[target].shift(1))
    features['LR(%)']= features['LR(%)']*100
    features['1PeriodVol(%)']= abs(features['LR(%)'])
    if LR== True: target_var= features['LR(%)']
    elif Vol== True: target_var= features['1PeriodVol(%)']
    else: target_var= target_var =features[target]
    features['y_future']= target_var.shift(-forecast_periods)
    features[['LR(%)','y_future','prices ethereum','1PeriodVol(%)']].tail(30).plot(subplots=True, figsize= (25, 25))
    FC= features.dropna(subset='y_future')
    FC = FC.map_partitions(impute).persist()
    func = lambda x: select_features(x.drop('y_future', axis=1), x['y_future'], fdr_level=p_value, hypotheses_independent=False, n_jobs=1)
    selected_features = FC.map_partitions(func, meta=FC, enforce_metadata=False).compute().columns.tolist()
    selected_features.append('y_future')
    if target_var.columns[0] != target: selected_features.append(target)
    if target_var.columns[0] != 'LR(%)': selected_features.append('LR(%)')
    if target_var.columns[0] != '1PeriodVol(%)':selected_features.append('1PeriodVol(%)')
    if len (selected_features) <= 5: print('not_enough_sig_feat')
    else: print('enough_sig_feat')
    final_features = FC[selected_features]
    return final_features

