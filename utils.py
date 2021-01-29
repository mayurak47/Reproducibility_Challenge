import torch
import math
from sklearn.preprocessing import MinMaxScaler
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from optimizers.laprop import LaProp
from torch.distributions import Normal
import numpy as np
import os
import random

def ensure_reproducibility(seed):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def snake_func(x, a=1):
    return x + torch.square(torch.sin(a * x))/a

def step(val):
    if math.floor(val)%2 == 1:
        return 0
    return 1

def gen_func_dataset(lower_limit, upper_limit, func, gap=0.01):
    inputs = torch.arange(lower_limit, upper_limit, gap).reshape(-1, 1)
    if func == "identity":
        outputs = inputs
    elif func == "tanh":
        outputs = torch.tanh(inputs)
    elif func == "sin":
        outputs = torch.sin(inputs)
    elif func == "square":
        outputs = torch.square(inputs)
    elif func == "step":
        outputs = torch.Tensor([step(i) for i in inputs]).reshape(-1, 1)
    elif func == "sinusoid":
        outputs = torch.sin(inputs)+torch.sin(4*inputs)/4
    else:
        raise Exception("Unknown function")
    return inputs, outputs

def sample_func_dataset(dataset_inputs, dataset_outputs, lower_lim, upper_lim, num_points):
    valid_indices = torch.where((dataset_inputs>=lower_lim) & (dataset_inputs<=upper_lim))[0]

    sampled_indices = valid_indices[torch.randperm(len(valid_indices))[:num_points]]

    sampled_inputs = dataset_inputs[sampled_indices]
    sampled_outputs = dataset_outputs[sampled_indices]


    return sampled_inputs, sampled_outputs

def normalize(sampled_inputs, function_inputs):
    scaler = MinMaxScaler()

    normalized_sample_inputs = torch.Tensor(scaler.fit_transform(sampled_inputs))
    normalized_function_inputs = torch.Tensor(scaler.transform(function_inputs))

    return normalized_sample_inputs, normalized_function_inputs

def train_extrapolation(model, train_inputs, train_outputs, epochs, verbose=False):

    train_losses = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)

    for epoch in tqdm(range(1, epochs+1), disable = not verbose):

        preds = model(train_inputs)
        loss = criterion(preds, train_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())
        if verbose:
            print(f"Loss on epoch {epoch}= {loss.item()}")

    return model, train_losses


def train_mnist(model, train_dataloader, epochs, verbose=False):
    train_losses = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-2)

    for epoch in tqdm(range(1, epochs+1), disable = not verbose):

        train_loss_total = 0
        num_steps = 0

        for i, (X, y) in enumerate(train_dataloader):
            X = X.to(device)
            y = y.to(device)
            preds = model(X)
            loss = criterion(preds, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_total += loss.item()
            num_steps += 1

        train_losses.append(train_loss_total/num_steps)

        if verbose:
            print(f"Loss on epoch {epoch} = {train_losses[-1]}")

    return model, train_losses

def train_cifar(model, train_dataloader, test_dataloader, epochs, verbose=False):
    train_losses = []
    test_losses = []

    train_accuracies = []
    test_accuracies = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = LaProp(model.parameters(), lr=4e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//2, gamma=0.1)

    for epoch in tqdm(range(1, epochs+1), disable = not verbose):

        model.train()
        train_loss_total = 0.0
        num_steps = 0
        correct = 0
        ### Train
        for i, batch in enumerate(train_dataloader):
            X, y = batch[0].to(device), batch[1].to(device)
            train_preds = model(X)

            loss = criterion(train_preds, y)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1

            train_preds = torch.max(train_preds, 1)[1]
            correct += (train_preds == y).float().sum()


        train_loss_total_avg = train_loss_total / num_steps
        train_accuracy = correct/(len(train_dataloader)*train_dataloader.batch_size)
        train_accuracies.append(train_accuracy)
        train_losses.append(train_loss_total_avg)

        model.eval()
        test_loss_total = 0.0
        num_steps = 0
        correct = 0
        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                X, y = batch[0].to(device), batch[1].to(device)

                test_preds = model(X)
                loss = criterion(test_preds, y)
                test_loss_total += loss.item()
                test_preds = torch.max(test_preds, 1)[1]
                correct += (test_preds == y).float().sum()

            num_steps += 1

        test_loss_total_avg = test_loss_total / num_steps
        test_accuracy = correct/(len(test_dataloader)*test_dataloader.batch_size)
        test_accuracies.append(test_accuracy)
        test_losses.append(test_loss_total_avg)

        scheduler.step()

        if verbose:
            print(f"Train accuracy on epoch {epoch}= {train_accuracies[-1]}")
            print(f"Test accuracy on epoch {epoch}= {test_accuracies[-1]}")

    return model, test_accuracies

def train_atmospheric_financial(model, train_dataloader, test_dataloader, epochs, verbose=False):

    train_losses = []
    test_losses = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(1, epochs+1), disable = not verbose):

        model.train()
        train_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(train_dataloader):
            X, y = batch[0].to(device), batch[1].to(device)

            train_preds = model(X)

            loss = criterion(train_preds, y)
            train_loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            num_steps += 1


        train_loss_total_avg = train_loss_total / num_steps
        train_losses.append(train_loss_total_avg)

        model.eval()
        test_loss_total = 0.0
        num_steps = 0

        for i, batch in enumerate(test_dataloader):
            with torch.no_grad():
                X, y = batch[0].to(device), batch[1].to(device)

                test_preds = model(X)
                loss = criterion(test_preds, y)
                test_loss_total += loss.item()

            num_steps += 1

        test_loss_total_avg = test_loss_total / num_steps
        test_losses.append(test_loss_total_avg)

        if verbose:
            print(f"Train loss on epoch {epoch} = {train_losses[-1]}")
            print(f"Test loss on epoch {epoch} = {test_losses[-1]}")

    return model, train_losses, test_losses

def train_mlp_sin_noise(model, train_inputs, train_outputs, test_inputs, test_outputs, epochs, verbose=False):
    train_losses = []
    test_losses = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    train_inputs = train_inputs.to(device)
    train_outputs = train_outputs.to(device)
    test_inputs = train_inputs.to(device)
    test_outputs = train_outputs.to(device)

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in tqdm(range(1, epochs+1), disable = not verbose):

        preds = model(train_inputs)
        loss = criterion(preds, train_outputs)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

        with torch.no_grad():
            preds = model(test_inputs)
            loss = criterion(preds, test_outputs)
            test_losses.append(loss.item())

        if verbose:
            print(f"Train loss on epoch {epoch} = {loss.item()}")
            print(f"Test loss on epoch {epoch} = {loss.item()}")

    return model, train_losses, test_losses

def train_rnn_sin_noise(rnn, train_seqs, test_seqs, epochs, verbose=False):

    train_losses = []
    test_losses = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rnn.to(device)

    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(rnn.parameters(), lr=1e-3)

    for i in tqdm(range(epochs), disable = not verbose):
        loss = 0
        steps = 0
        for seq, labels in train_seqs:
            seq, labels = seq.to(device), labels.to(device)

            optimizer.zero_grad()
            rnn.hidden_cell = torch.zeros(1, 1, rnn.hidden_layer_size)

            y_pred = rnn(seq)

            single_loss = loss_function(y_pred, labels)
            single_loss.backward()

            loss += single_loss.item()
            steps += 1
            optimizer.step()

        train_losses.append(loss/steps)

        loss = 0
        steps = 0
        for seq, labels in test_seqs:
            seq, labels = seq.to(device), labels.to(device)
            with torch.no_grad():
                rnn.hidden_cell = torch.zeros(1, 1, rnn.hidden_layer_size)

                y_pred = rnn(seq)

                single_loss = loss_function(y_pred, labels)

                loss += single_loss.item()
                steps += 1

        test_losses.append(loss/steps)

    return rnn, train_losses, test_losses


def train_sinusoid(model, X, y, epochs, verbose=False):

    train_losses = []

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    X = X.to(device)
    y = y.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())


    for epoch in tqdm(range(1, epochs+1), disable = not verbose):
        model.train()
        train_preds = model(X)
        loss = criterion(train_preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if verbose:
            print("\nTrain loss: {:.4f}".format(loss.item()))
    return model, train_losses

def create_inout_sequences(input_data, window=10):
    sequences = []
    L = len(input_data)
    for i in range(L-window):
        input_seq = input_data[i:i+window]
        output = input_data[i+window:i+window+1]
        sequences.append((input_seq, output))
    return sequences


def sin_with_noise(low_bound=0, up_bound=300, step=1, train_split=200, sigma=0.0):
    X = torch.arange(low_bound, up_bound, step)
    y = torch.sin(X/10)
    dist = Normal(0, sigma)
    y_plus_eps = y[:train_split] + dist.sample((train_split,))
    return X, y, y_plus_eps
