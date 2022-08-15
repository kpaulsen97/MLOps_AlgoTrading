# -*- coding: utf-8 -*-
"""
Created on Mon Aug 15 14:42:19 2022

@author: Lenovo
"""

from torch import nn
import torch.nn.functional as F
import torch

'''
deep learning model. simple two hidden layer model.
'''


class MyAwesomeQuantModel(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.fc1 = nn.Linear(num_input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_output)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # make sure input tensor is flattened
        # x = x.view(x.shape[0], -1)

        x = x.float()
        x = self.quant(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        x = self.dequant(x)

        return x