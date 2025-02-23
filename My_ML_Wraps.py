import xgboost
from sklearn.metrics import r2_score
from xgboost import dask as dxgb
import cupy as cp
import optuna

def RGS_XGB_Pandas(selected_features, parameter_grid, number_cvs=5):
# GPU Accelerated XGBoost Using Random Gridsearch for Hyperparameter Tuning 
    if param_grid is None:
        param_grid= {
                    'num_parallel_tree': [1,3,5,10,20],
                    'learning_rate': [0.01,0.01,0.05,0.1,0.5,1],
                    'max_depth': [3,6,12,24,48],
                    'gamma':[0,0.01,0.05,0.1,0.5,1,5,25],
                    'min_child_weight':[0.5,1,3,5],
                    'subsample':[1,0.05, 0.1,0.25,0.5],
                    'sampling_method':['uniform', 'gradient_based'],
                    'colsample_bytree':[1, 0.1, 0.5, 1],
                    'grow_policy':['depthwise', 'lossguide']
                    }
    else: param_grid=parameter_grid
    tscv = TimeSeriesSplit(n_splits=number_cvs)
    X_train_cp = xgb.DMatrix(X_train)
    X_test_cp = cp.array(X_test) 
    y_train_cp = cp.array(y_train) 
    y_test_cp = cp.array(y_test)
    basemodel= xgb.XGBRegressor(eval_metric='mae', early_stopping_rounds=25, 
                            device= 'cuda',tree_method='hist', n_jobs=1, verbose=0)
    rgs= RandomizedSearchCV(estimator=basemodel, verbose= 25, n_jobs=1,
                            cv=tscv,param_distributions=param_grid, n_iter=20, refit=True).fit()

def Optuna_XGB_Dask(dtrain, n_trials, n_rounds, 
                    eval_metric='mae', tree_method='hist', parameter_grid=None, early_stopping_rounds=10):  # Add early stopping parameter
    def objective(trial):
        if parameter_grid is None:
            param_grid = {
                "verbosity": 1,
                "tree_method": tree_method,
                "eval_metric": eval_metric,
                "lambda": trial.suggest_float("lambda", 1e-8, 100.0, log=True),
                "alpha": trial.suggest_float("alpha", 1e-8, 100.0, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.2, 1.0),
                "max_depth": trial.suggest_int("max_depth", 2, 20),
                "min_child_weight": trial.suggest_float("min_child_weight", 1e-8, 100, log=True),
                "learning_rate": trial.suggest_float("learning_rate", 1e-8, 1.0, log=True),
                "gamma": trial.suggest_float("gamma", 1e-8, 1.0, log=True)
            }   
        else: param_grid = parameter_grid
        try:
            output = dxgb.train(
                None,
                param_grid,
                dtrain,
                num_boost_round=n_rounds,
                early_stopping_rounds=early_stopping_rounds,  # Add early stopping
                evals=[(dtrain, "train")]
            )
            return output["history"]["train"][eval_metric][-1]
        except Exception as e:
            print(f"Error in training: {str(e)}")
            return float('inf')
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, n_jobs=-1, show_progress_bar=True)
    return study