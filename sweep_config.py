
sweep_configs = {
    "xgboost": {
        'method': 'random',
        'metric': {'name': 'rmse', 'goal': 'minimize'},
        'parameters': {
            'eta': {'values': [0.01, 0.05, 0.1, 0.3]},
            'max_depth': {'values': [3, 5, 7, 10]},
            'subsample': {'values': [0.6, 0.8, 1.0]},
            'colsample_bytree': {'values': [0.6, 0.8, 1.0]},
            'gamma': {'values': [0, 0.1, 0.2]},
            'reg_lambda': {'values': [0, 1, 2]}, 
            'alpha': {'values': [0, 1, 2]}
        }
    },

    
    "lightgbm": {
        'method': 'random',
        'metric': {'name': 'rmse', 'goal': 'minimize'},
        'parameters': {
            'learning_rate': {'values': [0.01, 0.05, 0.1]},
            'num_leaves': {'values': [31, 50, 70, 100]},
            'max_depth': {'values': [-1, 10, 20, 30]},
            'min_data_in_leaf': {'values': [10, 20, 50, 100]},
            'feature_fraction': {'values': [0.6, 0.8, 1.0]},
            'bagging_fraction': {'values': [0.6, 0.8, 1.0]},
            'lambda_l1': {'values': [0, 0.1, 0.5]},
            'lambda_l2': {'values': [0, 0.1, 0.5]}
        }
    },
    "random_forest": {
        'method': 'random',
        'metric': {'name': 'rmse', 'goal': 'minimize'},
        'parameters': {
            'n_estimators': {'values': [100, 200, 500, 1000]},
            'max_depth': {'values': [5, 10, 15, 20, None]},
            'min_samples_split': {'values': [2, 5, 10]},
            'min_samples_leaf': {'values': [1, 2, 4]},
            'max_features': {'values': ['auto', 'sqrt', 'log2']}
        }
    },
    "catboost": {
        'method': 'random',
        'metric': {'name': 'rmse', 'goal': 'minimize'},
        'parameters': {
            'iterations': {'values': [100, 200, 500]},
            'depth': {'values': [6, 8, 10]},
            'learning_rate': {'values': [0.01, 0.05, 0.1]},
            'l2_leaf_reg': {'values': [1, 3, 5, 7]},
            'bagging_temperature': {'values': [0, 1, 2]}
        }
    }
}
