# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:45:14 2022

@author: Lenovo
"""

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#import wandb
#wandb.init()


from quant_model import MyAwesomeQuantModel

from omegaconf import OmegaConf
config = OmegaConf.load('src/models/config.yaml')

'''
Script designed to take the model, train it, and test it.
'''

data = torch.load(
    "data/processed/processed.pt"
)
x_train, x_test, y_train, y_test = data.values()
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)
num_input = len(x_train[0])

net = MyAwesomeQuantModel(num_input, 2)
optimizer = optim.SGD(net.parameters(), lr=config.hyperparameters.learning_rate)
criterion = nn.CrossEntropyLoss()
#wandb.watch(net)

# we could have done this ourselves,
# but we should be aware of sklearn and its tools
from sklearn.metrics import accuracy_score


# setting hyperparameters and gettings epoch sizes
batch_size = config.hyperparameters.batch_size
num_epochs = config.hyperparameters.num_epochs
num_samples_train = x_train.shape[0]
num_batches_train = num_samples_train // batch_size


# setting up lists for handling loss/accuracy
train_acc, train_loss = [], []
test_acc, test_loss = [], []
cur_loss = 0
losses = []

get_slice = lambda i, size: range(i * size, (i + 1) * size)

for epoch in range(num_epochs):
    # Forward -> Backprob -> Update params
    ## Train
    cur_loss = 0
    net.train()
    for i in range(num_batches_train):
        optimizer.zero_grad()
        slce = get_slice(i, batch_size)
        output = net(x_train[slce])

        # compute gradients given loss
        target_batch = y_train[slce]
        batch_loss = criterion(output, target_batch)
        batch_loss.backward()
        optimizer.step()

        cur_loss += batch_loss
    losses.append(cur_loss / batch_size)
    #wandb.log({'loss':cur_loss/batch_size})

    net.eval()
    ### Evaluate training
    train_preds, train_targs = [], []
    for i in range(num_batches_train):
        slce = get_slice(i, batch_size)
        output = net(x_train[slce])

        preds = torch.max(output, 1)[1]

        train_targs += list(y_train[slce].numpy())
        train_preds += list(preds.data.numpy())

    train_acc_cur = accuracy_score(train_targs, train_preds)

    train_acc.append(train_acc_cur)

    if epoch % 10 == 0:
        print(
            "Epoch %2i : Train Loss %f , Train acc %f"
            % (epoch + 1, losses[-1], train_acc_cur)
        )

# Test model

output = net(x_test)
preds = torch.max(output, 1)[1]
test_targs = y_test.numpy()
test_preds = preds.data.numpy()
test_acc_cur = accuracy_score(test_targs, test_preds)
print("Test accuracy for float model: ", test_acc_cur)

net_quant = torch.quantization.convert(net)
output = net_quant(x_test)
preds = torch.max(output, 1)[1]
test_targs = y_test.numpy()
test_preds = preds.data.numpy()
test_acc_cur = accuracy_score(test_targs, test_preds)
print("Test accuracy for quant model: ", test_acc_cur)


epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, "r")
plt.legend(["Train Accucarcy"])
plt.xlabel("Updates"), plt.ylabel("Acc")
