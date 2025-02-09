# Imports
import pandas as pd
import datetime as dt
from tsfresh import extract_features
from tsfresh.feature_extraction import EfficientFCParameters, ComprehensiveFCParameters, MinimalFCParameters
from tsfresh.utilities.dataframe_functions  import roll_time_series
import numpy as np
from tsfresh.utilities.dataframe_functions import impute
import dask.dataframe as dd
from tsfresh.utilities.distribution import ClusterDaskDistributor
from tsfresh.convenience.bindings import dask_feature_extraction_on_chunk
import multiprocessing

# Provides distributed and local feature extraction functionality for cryptocurrency data
# using TsFresh. Supports both Pandas and Dask DataFrame processing. 
# Dependencies:
#     - pandas
#     - tsfresh
#     - dask
#     - numpy
#     - multiprocessing
# Required Data Format:
#     Input DataFrame columns: ['datetime', 'variable', 'value', 'id']
#     The same format as the output of my API_wraps.py functions.


# 1.Feature extraction for Dask DataFrame with Dask- Recommended for large datasets
#     Args:
#         df (dask.DataFrame): Input DataFrame in long format
#         ParameterComplexity (int): Feature complexity level
#             0: Minimal features
#             1: Efficient features (default)
#             2: Comprehensive features
#     Returns:
#         dask.DataFrame: Extracted features with datetime index
#     Notes:
#         - Optimized for distributed processing
#         - Uses partition-wise rolling calculations
def EF_Dask(df, ParameterComplexity=1, splits=5, target='prices ethereum', LR=False, Vol=False, forecast_periods=1,):
    if ParameterComplexity ==0: FC_parameters= MinimalFCParameters()
    elif ParameterComplexity ==1: FC_parameters= EfficientFCParameters()
    else: FC_parameters= ComprehensiveFCParameters()
    meta= {'datetime': 'string', 'variable': 'string', 'value': 'float64', 'id': 'string'}
    func=  lambda x: roll_time_series(x, column_id='variable', column_sort='datetime', n_jobs=1, chunksize=None).ffill()
    rolled= df.map_partitions(func, meta = meta)
    rolled= rolled.astype(meta)
    dates =rolled[['datetime','id']].groupby('id').last()
    concat = rolled[['variable','value', 'datetime','id']].groupby('id').last().reset_index(drop=False)
    rolled_grouped = rolled.groupby(['id','variable'])
    features= dask_feature_extraction_on_chunk(rolled_grouped, column_id='id', column_sort='datetime',
                                                column_kind='variable', column_value='value', default_fc_parameters= FC_parameters)
    features= features.reset_index(drop=True).astype({'variable': 'string', 'value': 'float64', 'id': 'string'})
    features= features.join(dates, how='left', on='id')
    features= dd.concat([features, concat], axis=0).drop('id', axis=1)
    features= features.categorize('variable')
    features= features.pivot_table(index='datetime', columns= 'variable', values='value')
    features['LR(%)']= np.log(features[target])- np.log(features[target].shift(1))
    features['LR(%)']= features['LR(%)']*100
    features['1PeriodVol(%)']= abs(features['LR(%)'])
    if LR== True: target_var= features['LR(%)']
    elif Vol== True: target_var= features['1PeriodVol(%)']
    else: target_var= features[target]
    features['y_future']= target_var.shift(-forecast_periods)
    features=features.repartition(npartitions=splits)
    return features

# 2.Feature extraction using local multiprocessing for Pandas DataFrame. Simplest for small datasets
    # Inputs:
    #     df (pandas.DataFrame): Input DataFrame in long format with columns:
    #     ParameterComplexity (int): Feature extraction complexity
    #         0: Minimal (fastest, fewest features)
    #         1: Efficient (default, balanced)
    #         2: Comprehensive (slowest, all features)
    # Returns:
    #     pandas.DataFrame: Extracted features with datetime index
    # Notes:
    #     - Uses local CPU cores for parallel processing
    #     - Memory efficient for medium-sized datasets
def EF_Pandas_MultiprocessingDistributor(df, ParameterComplexity=1):
    if ParameterComplexity ==0: FC_parameters= MinimalFCParameters()
    elif ParameterComplexity ==1: FC_parameters= EfficientFCParameters()
    else: FC_parameters= ComprehensiveFCParameters()
    feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime',
                            'column_value':'value','default_fc_parameters': FC_parameters}
    rolled = roll_time_series(df, column_id='variable', column_sort='datetime', n_jobs=14).ffill()
    dates= rolled.groupby('id').last()['datetime']
    concat= rolled.groupby('id').last().reset_index(drop=False)
    raw_features = extract_features(rolled, **feature_extrac_params, n_jobs=14, pivot=False)
    features= pd.DataFrame(raw_features, columns=['id','variable','value'])
    features= features.join(dates, how='left', on='id')
    features= pd.concat([features,concat], axis=0)
    features= features.pivot_table(index="datetime", columns="variable", values="value")
    return features
                                                        
# 3.Distributed feature extraction using Dask scheduler for Pandas DataFrame. -May have best performance for mid-size datasets that fit in memory
    # Inputs:
    #     df (pandas.DataFrame): Input DataFrame in long format
    #     scheduler_address (str): Dask scheduler address
    #     ParameterComplexity (int): Feature complexity level
    #         0: Minimal features
    #         1: Efficient features (default)
    #         2: Comprehensive features
    # Returns:
    #     pandas.DataFrame: Extracted features with datetime index
def EF_Pandas_DaskDistributor(df, scheduler_address, ParameterComplexity=1):
    if ParameterComplexity ==0: FC_parameters= MinimalFCParameters()
    elif ParameterComplexity ==1: FC_parameters= EfficientFCParameters()
    else: FC_parameters= ComprehensiveFCParameters()
    feature_extrac_params= {'column_id':'id', 'column_kind':'variable', 'column_sort':'datetime',
                        'column_value':'value','default_fc_parameters': FC_parameters}
    rolled=roll_time_series(df, distributor=ClusterDaskDistributor(scheduler_address), column_id='variable', column_sort='datetime').ffill()
    dates= rolled.groupby('id').last()['datetime']
    concat= rolled.groupby('id').last().reset_index(drop=False)
    raw_features= extract_features(rolled,distributor=ClusterDaskDistributor(scheduler_address), **feature_extrac_params, pivot=False)
    features= pd.DataFrame(raw_features, columns=['id','variable','value'])
    features= features.join(dates, how='left', on='id')
    features= pd.concat([features,concat], axis=0)
    features= features.pivot_table(index="datetime", columns="variable", values="value")
    return features                  
