# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:34:48 2022

@author: Lenovo
"""

#ridicolously long times.

import torch
import torch.optim as optim
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

#import wandb
#wandb.init()


from model import MyAwesomeModel

from omegaconf import OmegaConf
config = OmegaConf.load('C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\src\\models\\config.yaml')

'''
Script designed to take the model, train it, and test it.
'''
from torch.utils.data import DataLoader, Dataset

class DDP(Dataset):
    def __init__(self, path_to_folder: str, key) -> None:
        
        self.data = torch.load(path_to_folder)[key]
        
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index: int) -> torch.Tensor:
        
        return self.data[index]
    

if __name__ == '__main__':
    
    data = torch.load(
        "C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\data/processed/processed.pt"
    )
    x_train, x_test, y_train, y_test = data.values()
    y_train = torch.from_numpy(y_train.values)
    y_test = torch.from_numpy(y_test.values)
    num_input = len(x_train[0])
    num_samples_train = x_train.shape[0]
    
    x_train = DDP("C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\data/processed/processed.pt", "x_train")
    x_train = DataLoader(
        x_train,
        batch_size = config.hyperparameters.batch_size,
        num_workers= 1  # this is the number of threds we want to parallize workload over
    )
    
    
    
    net = MyAwesomeModel(num_input, 2)
    optimizer = optim.SGD(net.parameters(), lr=config.hyperparameters.learning_rate)
    criterion = nn.CrossEntropyLoss()
    #wandb.watch(net)
    
    # we could have done this ourselves,
    # but we should be aware of sklearn and its tools
    from sklearn.metrics import accuracy_score
    
    
    # setting hyperparameters and gettings epoch sizes
    batch_size = config.hyperparameters.batch_size
    num_epochs = config.hyperparameters.num_epochs
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
        for batch_idx, x_batch in enumerate(x_train):
            if(batch_idx == num_batches_train):
                continue
            optimizer.zero_grad()
            slce = get_slice(batch_idx, config.hyperparameters.batch_size)
            output = net(x_batch)
    
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
        for batch_idx, x_batch in enumerate(x_train):
            if(batch_idx == num_batches_train):
                continue
            slce = get_slice(batch_idx, config.hyperparameters.batch_size)
            output = net(x_batch)
    
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
    print("Test accuracy: ", test_acc_cur)
    
    
    epoch = np.arange(len(train_acc))
    plt.figure()
    plt.plot(epoch, train_acc, "r")
    plt.legend(["Train Accucarcy"])
    plt.xlabel("Updates"), plt.ylabel("Acc")
