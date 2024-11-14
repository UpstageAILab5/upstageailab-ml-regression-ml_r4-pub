
method = 'bayes'
metric = 'rmse'

sweep_config_baseline = {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            "model": {"values": ["xgboost", "random_forest"]},#, "lightgbm", "catboost"]},
            "dataset_name": {"values": ["baseline"]},#, "encoded", "feat_null-preped_freq-encoded", "null-preped_freq-encoded"]},
            "features": {"values": ["baseline", "removed", "minimum", "medium"]},
            "categorical_encoding": {"values": ["baseline", "freq"]},
            "outlier_removal": {"values": ["baseline", "none", "iqr_modified"]},
            "split_type": {"values": ["holdout", "kfold"]},
            'xgboost_n_estimators': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 1000
            },
            'xgboost_eta': {
                'distribution': 'uniform',  # log_uniform 대신 uniform 사용
                'min': 0.01,
                'max': 0.3
            },
            'xgboost_max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'xgboost_subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'xgboost_colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'xgboost_gamma': {  # xgboost-specific
                'distribution': 'uniform',
                'min': 0.0,
                'max': 5.0
            },
            'xgboost_reg_lambda': {
                'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
            },
            'xgboost_alpha': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
        
        }
}

sweep_config_xgb = {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            "model": {"values": ["xgboost"]},
            "dataset_name": {"values": ["baseline"]},#, "encoded", "feat_null-preped_freq-encoded", "null-preped_freq-encoded"]},
            "features": {"values": ["baseline", "removed", "minimum", "medium"]},
            "split_type": {"values": ["holdout", "kfold"]},
            'xgboost_n_estimators': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 1000
            },
            'xgboost_eta': {
                'distribution': 'uniform',  # log_uniform 대신 uniform 사용
                'min': 0.01,
                'max': 0.3
            },
            'xgboost_max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'xgboost_subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'xgboost_colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'xgboost_gamma': {  # xgboost-specific
                'distribution': 'uniform',
                'min': 0.0,
                'max': 5.0
            },
            'xgboost_reg_lambda': {
                'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
            },
            'xgboost_alpha': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
        
        }
}


sweep_config_lightgbm = {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            "model": {"values": ["lightgbm"]},
            "dataset_name": {"values": ["baseline", "encoded", "feat_null-preped_freq-encoded", "null-preped_freq-encoded"]},
            "features": {"values": ["baseline", "removed", "minimum", "medium"]},
            "split_type": {"values": ["kfold", "holdout"]},
    'lightgbm_n_estimators': {
        'distribution': 'int_uniform',
        'min': 100,
        'max': 1000
            },
            'lightgbm_learning_rate': {
                'distribution': 'uniform',  # log_uniform 대신 uniform 사용
                'min': 0.01,
                'max': 0.3
                },
            'lightgbm_colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lightgbm_subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'lightgbm_num_leaves': {
                'distribution': 'int_uniform',
                'min': 31,
                'max': 255
            },
         
            'lightgbm_max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'lightgbm_min_data_in_leaf': {  # lightgbm-specific
                'distribution': 'int_uniform',
                'min': 10,
                'max': 100
            },
            'lightgbm_feature_fraction': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lightgbm_bagging_fraction': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lightgbm_lambda_l1': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
            'lightgbm_lambda_l2': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
            }
}
sweep_configs = {
        'method': method,
        'metric': {'name': metric, 'goal': 'minimize'},
        'parameters': {
            "model": {"values": ["xgboost", "random_forest", "lightgbm", "catboost"]},
            "dataset_name": {"values": ["baseline", "encoded", "feat_null-preped_freq-encoded", "null-preped_freq-encoded"]},
            "features": {"values": ["baseline", "removed", "minimum", "medium"]},
            "split_type": {"values": ["kfold", "holdout"]},
            'xgboost_n_estimators': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 1000
            },
            'xgboost_eta': {
                'distribution': 'uniform',  # log_uniform 대신 uniform 사용
                'min': 0.01,
                'max': 0.3
            },
            'xgboost_max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'xgboost_subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'xgboost_colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'xgboost_gamma': {  # xgboost-specific
                'distribution': 'uniform',
                'min': 0.0,
                'max': 5.0
            },
            'xgboost_reg_lambda': {
                'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
            },
            'xgboost_alpha': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
          

    'lightgbm_n_estimators': {
                'distribution': 'int_uniform',
                'min': 100,
                'max': 1000
            },
            'lightgbm_learning_rate': {
                'distribution': 'uniform',  # log_uniform 대신 uniform 사용
                'min': 0.01,
                'max': 0.3
                },
            'lightgbm_colsample_bytree': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lightgbm_subsample': {
                'distribution': 'uniform',
                'min': 0.6,
                'max': 1.0
            },
            'lightgbm_num_leaves': {'values': [31, 50, 70, 100]},
            'lightgbm_max_depth': {
                'distribution': 'int_uniform',
                'min': 3,
                'max': 10
            },
            'lightgbm_min_data_in_leaf': {  # lightgbm-specific
                'distribution': 'int_uniform',
                'min': 10,
                'max': 100
            },
            'lightgbm_feature_fraction': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lightgbm_bagging_fraction': {
                'distribution': 'uniform',
                'min': 0.4,
                'max': 1.0
            },
            'lightgbm_lambda_l1': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
            'lightgbm_lambda_l2': {
                'distribution': 'uniform',
                'min': 0.0,
                'max': 10.0
            },
            
    


            'random_forest_n_estimators': {
            'distribution': 'int_uniform',
            'min': 50,
            'max': 300
        },
        'random_forest_max_depth': {
            'distribution': 'int_uniform',
            'min': 5,
            'max': 15
        },
        'random_forest_min_samples_split': {
            'distribution': 'int_uniform',
            'min': 2,
            'max': 10
        },
        'random_forest_min_samples_leaf': {
            'distribution': 'int_uniform',
            'min': 1,
            'max': 5
        },
        'random_forest_max_features': {
            'values': ['sqrt', 'log2']
        },
        'random_forest_bootstrap': {
            'values': [True]
        },


            'catboost_iterations': {
            'distribution': 'int_uniform',
            'min': 100,
            'max': 1000
        },
        'catboost_depth': {
            'distribution': 'int_uniform',
            'min': 4,
            'max': 10
        },
        'catboost_learning_rate': {
            'distribution': 'uniform',  # log_uniform 대신 uniform 사용
            'min': 0.01,
            'max': 0.3
        },
        'catboost_l2_leaf_reg': {
            'distribution': 'uniform',
            'min': 1.0,
            'max': 10.0
        },
        'catboost_bagging_temperature': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 1.0
        },
        'catboost_random_strength': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 10.0
        },
        'catboost_border_count': {
            'distribution': 'int_uniform',
            'min': 32,
            'max': 255
        },
        'catboost_boosting_type': {
            'values': ['Ordered', 'Plain']
        
        }
    }
}