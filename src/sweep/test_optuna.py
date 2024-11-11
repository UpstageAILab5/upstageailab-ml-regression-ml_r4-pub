import optuna
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split

# Load configuration from YAML
def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def objective(trial, model_name, X, y, config):
    params = {}
    # Load parameter space from config
    for param, param_config in config[model_name].items():
        distribution = param_config['distribution']
        if distribution == 'int_uniform':
            params[param] = trial.suggest_int(param, param_config['min'], param_config['max'])
        elif distribution == 'log_uniform':
            params[param] = trial.suggest_loguniform(param, param_config['min'], param_config['max'])
        elif distribution == 'uniform':
            params[param] = trial.suggest_uniform(param, param_config['min'], param_config['max'])
        else:
            raise ValueError(f"Unsupported distribution type: {distribution}")

    # Model selection
    if model_name == 'xgboost':
        model_class = xgb.XGBRegressor
    elif model_name == 'lightgbm':
        model_class = lgb.LGBMRegressor
    else:
        raise ValueError("Unsupported model name. Use 'xgboost' or 'lightgbm'.")

    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    mse_scores = []
    for train_index, val_index in kf.split(X):
        X_train, X_val = X[train_index], X[val_index]
        y_train, y_val = y[train_index], y[val_index]

        # Model training
        model = model_class(**params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        mse_scores.append(mse)

    return np.mean(mse_scores)

# Function to run the optimization and return the best parameters
def optimize_model(model_name, X, y, config_path='config.yaml'):
    config = load_config(config_path)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, model_name, X, y, config), n_trials=50)
    return study.best_params

# Function to train the final model with the best parameters and evaluate on the test set
def train_and_evaluate(model_name, X_train, y_train, X_test, y_test, best_params):
    if model_name == 'xgboost':
        model_class = xgb.XGBRegressor
    elif model_name == 'lightgbm':
        model_class = lgb.LGBMRegressor
    else:
        raise ValueError("Unsupported model name. Use 'xgboost' or 'lightgbm'.")

    # Train the final model
    model = model_class(**best_params)
    model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Evaluate the performance
    mse = mean_squared_error(y_test, y_pred)
    print(f"Test MSE for {model_name}: {mse}")

    # Save the predictions
    np.savetxt(f"{model_name}_predictions.txt", y_pred, delimiter=",")
    print(f"Predictions saved to {model_name}_predictions.txt")

# Data preparation (example using Boston dataset)
data = load_boston()
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Run optimization and training for XGBoost
xgb_best_params = optimize_model('xgboost', X_train, y_train)
train_and_evaluate('xgboost', X_train, y_train, X_test, y_test, xgb_best_params)

# Run optimization and training for LightGBM
lgb_best_params = optimize_model('lightgbm', X_train, y_train)
train_and_evaluate('lightgbm', X_train, y_train, X_test, y_test, lgb_best_params)
