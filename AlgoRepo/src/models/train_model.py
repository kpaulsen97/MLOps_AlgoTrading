import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append('C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\src\\models')

from model import MyAwesomeModel


data = torch.load('C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\data\\processed\\processed.pt')
x_train, x_test, y_train, y_test = data.values()
y_train = torch.from_numpy(y_train.values)
y_test = torch.from_numpy(y_test.values)
num_input = len(x_train[0])

net = MyAwesomeModel(num_input, 2)
optimizer = optim.SGD(net.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# we could have done this ourselves,
# but we should be aware of sklearn and its tools
from sklearn.metrics import accuracy_score


# setting hyperparameters and gettings epoch sizes
batch_size = 20
num_epochs = 400
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
        print("Epoch %2i : Train Loss %f , Train acc %f" % (
                epoch+1, losses[-1], train_acc_cur))

epoch = np.arange(len(train_acc))
plt.figure()
plt.plot(epoch, train_acc, 'r')
plt.legend(['Train Accucarcy'])
plt.xlabel('Updates'), plt.ylabel('Acc')
