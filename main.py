import pandas as pd
import torch as pt
from torch import optim
from helpers import *
import numpy as np

file_path = 'data/train_features.csv'
file_path_labels = 'data/train_targets_scored.csv'

n_rows = 100
data = pd.read_csv(file_path, nrows=n_rows)
data_labels = pd.read_csv(file_path_labels, nrows=n_rows)

# print(data.columns) # 876 columns (ingluding sig_id)
# print(data_labels.columns) # 207 columns (including sig_id)

train_col_count = len(data.columns) - 1
# train_col_count = 103
train_n = data.shape[0]
label_col_count = len(data_labels.columns) - 1
label_n = data_labels.shape[0]

# handle categorical values (convert them into numerical)
data['cp_type'] = category2codes(data, 'cp_type')
data['cp_dose'] = category2codes(data, 'cp_dose')

# concatenate groundtruth
data = pd.merge(data, data_labels, on='sig_id', how='right')
# print(data_full.columns) # 1082 columns (including sig_id)

# remove ID column
data = data.drop('sig_id', axis=1)

# prepare input and output for the training
input = data[data.columns[:train_col_count]].to_numpy().astype(np.single) # has to be Float32, otherwise PyTorch is complaining for getting a Double (Float64)
input = pt.from_numpy(input)
target = data[data.columns[-label_col_count:]].to_numpy().astype(np.single) # take last entries, negative sign
target = pt.from_numpy(target).squeeze(0)

### constructing the network
network = MyNet(input, target)

# construct the optimizer
op = optim.SGD(network.parameters(), 0.01)
criterion = pt.nn.CrossEntropyLoss()

n_iterations = 15000
for i in range(n_iterations):
    op.zero_grad()
    output = network(input)
    L = pt.max(target, 1)[1] # convert one-hot encodings into the label
    loss = criterion(output, L)
    print("Iteration " + str(i) + ", loss: " + str(loss.item()))
    loss.backward()
    op.step()

op.zero_grad()
output = network(input)

criterion = pt.nn.CrossEntropyLoss()
loss = criterion(output, pt.max(target, 1)[1])
print("Loss after: " + str(loss.item()))
