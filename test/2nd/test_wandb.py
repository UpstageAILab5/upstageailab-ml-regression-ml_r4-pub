import wandb
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

def train_model(model, X_train, y_train, X_test, y_test, config):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    wandb.log({"rmse": rmse})
    return rmse

def prepare_data(X, y, split_method="holdout"):
    if split_method == "holdout":
        return train_test_split(X, y, test_size=0.2, random_state=42)
    elif split_method == "kfold":
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        return kf.split(X)

def run_experiment(models, dataset_split_methods, features, scaling_methods, null_prep_methods, X, y):
    for model in models:
        for split_method in dataset_split_methods:
            for feature_set in features:
                for scale_method in scaling_methods:
                    for null_method in null_prep_methods:
                        config = {
                            "model": str(model),
                            "split_method": split_method,
                            "features": feature_set,
                            "scaling": scale_method,
                            "null_prep": null_method
                        }

                        wandb.init(project="ml_experiment", config=config)

                        # Data preparation
                        X_prepared = feature_preparation(X, feature_set)  # Customize this function as needed
                        y_prepared = y.copy()

                        # Null preparation - example placeholder
                        if null_method == "drop":
                            X_prepared = X_prepared.dropna()

                        # Scaling - example placeholder
                        if scale_method == "standard":
                            scaler = StandardScaler()
                            X_prepared = scaler.fit_transform(X_prepared)

                        if split_method == "holdout":
                            X_train, X_test, y_train, y_test = prepare_data(X_prepared, y_prepared, "holdout")
                            train_model(model, X_train, y_train, X_test, y_test, config)
                        elif split_method == "kfold":
                            for train_index, test_index in prepare_data(X_prepared, y_prepared, "kfold"):
                                X_train, X_test = X_prepared[train_index], X_prepared[test_index]
                                y_train, y_test = y_prepared[train_index], y_prepared[test_index]
                                train_model(model, X_train, y_train, X_test, y_test, config)

                        wandb.finish()

def feature_preparation(X, feature_set):
    # Example feature selection logic - to be customized
    if feature_set == "base":
        return X
    elif feature_set == "apt":
        return X[['feature1', 'feature2']]  # Adjust as necessary
    return X

models = ['random_forest', 'xgboost', 'lightgbm']
dataset_split_methods = ['holdout', 'kfold']
features = ['base', 'apt']
scaling_methods = ['standard', 'robust', 'minmax']
null_prep_methods = ['drop', 'fill']
run_experiment(models, dataset_split_methods, features, scaling_methods, null_prep_methods, X, y)