import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

def preprocess_financial_data(file_path):
    df = pd.read_csv(file_path, header=1, dtype={'Index Dates': int, 'Index Values': float}, skipfooter=1)

    # return df[df['Index Dates'] <= 20200531]
    y_train = df[df['Index Dates'] <= 20200131]["Index Values"].values
    y_test = df[(df['Index Dates'] > 20200131) & (df['Index Dates'] <= 20200531)]["Index Values"].values
    
    X_train = np.array(list(range(len(y_train))))
    X_test = np.array(list(range(len(y_train), len(y_train)+len(y_test))))

    # print(X_train)
    # print(X_test)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train.reshape(-1, 1))
    X_test = scaler.transform(X_test.reshape(-1, 1))

    y_train = torch.Tensor(y_train).reshape(-1, 1)
    y_test = torch.Tensor(y_test).reshape(-1, 1)
    X_train = torch.Tensor(X_train).reshape(-1, 1)
    X_test = torch.Tensor(X_test).reshape(-1, 1)

    # print(y_train)
    # print(y_test)
    # print(X_train)
    # print(X_test)
    return (X_train, y_train), (X_test, y_test)

class FinancialDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def financial_dataloaders(X_train, y_train, X_test, y_test):
    # df = preprocess_financial_data("data/IndexHistory_19950101.csv")

    # y_train = df[df['Index Dates'] <= 20200131]["Index Values"].values
    # y_test = df[df['Index Dates'] > 20200131]["Index Values"].values  

    # X_train = np.array(list(range(len(y_train))))
    # X_test = np.array(list(range(len(y_train), len(y_train)+len(y_test))))

    # scaler = MinMaxScaler(feature_range=(0, 1))
    # X_train = scaler.fit_transform(X_train.reshape(-1, 1))
    # X_test = scaler.transform(X_test.reshape(-1, 1))



    train_ds = FinancialDataset(X_train, y_train)
    test_ds = FinancialDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    return train_dl, test_dl
