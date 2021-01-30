import json
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

'''data in 10_year_temp_data.json downloaded from https://join.fz-juelich.de/access/'''

'''find average weekly dataset, removing outliers'''
def preprocess_data(file_path):
    with open(file_path) as f:
        json_data = json.load(f)
    
    temp = json_data['mean']

    weekly_temp = []
    for i in range(0, len(temp), 7):
        curr_week_mean = sum(temp[i:i+7])/7
        if curr_week_mean < 18:
            continue
        weekly_temp.append(curr_week_mean)

    return weekly_temp

class TempDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.Tensor(X).reshape(-1, 1)
        self.y = torch.Tensor(y).reshape(-1, 1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


'''train-test split'''
def atmospheric_dataloaders():
    data = preprocess_data("data/10_year_temp_data.json")
    y_train = data[:312]
    y_test = data[312:]

    X_train = list(range(len(y_train)))
    X_test = list(range(len(y_test)))


    train_ds = TempDataset(X_train, y_train)
    test_ds = TempDataset(X_test, y_test)

    train_dl = DataLoader(train_ds, batch_size=64, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=64, shuffle=False)

    return train_dl, test_dl