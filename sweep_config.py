
method = 'bayes'
metric = 'rmse'

sweep_configs = {
    "xgboost": {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            'n_estimators': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 1000
            },
            'eta': {
                'distribution': 'log_uniform',
                'min': 0.01,
                'max': 0.3
            },
            'eta': {'values': [0.01, 0.05, 0.1, 0.3]},
            'max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'gamma': {  # xgboost-specific
            'distribution': 'uniform',
            'min': 0.0,
            'max': 5.0
            },
            'reg_lambda': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
            },
            'alpha': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
            }
        }
    },

    
    "lightgbm": {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            'n_estimators': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 1000
            },
            'learning_rate': {
                'distribution': 'log_uniform',
                'min': 0.01,
                'max': 0.3
                },
            'colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'num_leaves': {'values': [31, 50, 70, 100]},
            'max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'min_data_in_leaf': {  # lightgbm-specific
                'distribution': 'int_uniform',
                'min': 10,
                'max': 100
            },
            'feature_fraction': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'bagging_fraction': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lambda_l1': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
            'lambda_l2': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            }
        }
    },
    "random_forest": {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            'n_estimators': {
            'distribution': 'int_uniform',
            'min': 50,
            'max': 300
        },
        'max_depth': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 15
        },
        'min_samples_split': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 10
        },
        'min_samples_leaf': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5
        },
        'max_features': {
            'values': ['sqrt', 'log2']
        },
        'bootstrap': {
            'values': [True]
        }
    
        }
    },
    "catboost": {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            'iterations': {
            'distribution': 'int_uniform',
            'min': 100,
            'max': 1000
        },
        'depth': {
            'distribution': 'int_uniform',
            'min': 4,
            'max': 10
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 0.01,
            'max': 0.3
        },
        'l2_leaf_reg': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 10.0
        },
        'bagging_temperature': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'random_strength': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
        },
        'border_count': {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 255
        },
        'boosting_type': {
            'values': ['Ordered', 'Plain']
        
        }
    }
}
}