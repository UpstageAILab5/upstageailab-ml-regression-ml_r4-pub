import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self, null_prep="mean", scaling="standard", encoding="none"):
        self.null_prep = null_prep
        self.scaling = scaling
        self.encoding = encoding
        self.scaler = None
        self.imputer = None
        self.encoder = None

    def handle_nulls(self, X):
        if self.null_prep == "mean":
            self.imputer = SimpleImputer(strategy="mean")
            return self.imputer.fit_transform(X)
        elif self.null_prep == "median":
            self.imputer = SimpleImputer(strategy="median")
            return self.imputer.fit_transform(X)
        elif self.null_prep == "drop":
            return X.dropna()
        return X

    def scale_data(self, X):
        if self.scaling == "standard":
            self.scaler = StandardScaler()
            return self.scaler.fit_transform(X)
        elif self.scaling == "minmax":
            self.scaler = MinMaxScaler()
            return self.scaler.fit_transform(X)
        return X

    def encode_data(self, X):
        if self.encoding == "onehot":
            self.encoder = OneHotEncoder()
            return self.encoder.fit_transform(X).toarray()
        return X

    def preprocess(self, X):
        X = self.handle_nulls(X)
        X = self.scale_data(X)
        X = self.encode_data(X)
        return X
