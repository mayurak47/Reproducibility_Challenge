import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

'''data provided by authors; performance is tested for a number of days beyond the given data'''
def get_temp_data():
    time = [0, 9, 24+15, 24+23, 48+9, 48+15, 48+23, 72+9, 72+17, 72+23, 96+9, 96+23, 120+8, 120+10, 120+17, 120+23, 144+8, 144+18, 168+9, 168+23, 192+8, 192+18, 192+23, 216+9, 216+14]
    temp = [35.95, 36.50, 36.60, 36.17, 36.50, 37.09, 37.24, 36.53, 37.25, 36.79, 36.59, 
        36.62, 36.45, 36.8, 36.99, 36.19, 36.4, 37.01, 36.5, 35.7, 36.4, 36.98, 36.59, 36.65, 36.87]

    X_train = np.array(time).reshape(-1, 1)
    y_train = torch.Tensor(temp).reshape(-1, 1)
    X_test = np.arange(240, 700, 10.).reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = torch.Tensor(scaler.fit_transform(X_train.reshape(-1, 1)))
    X_test = torch.Tensor(scaler.transform(X_test.reshape(-1, 1)))

    return X_train, y_train, X_test

