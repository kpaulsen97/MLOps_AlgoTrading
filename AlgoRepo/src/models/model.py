# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:41:15 2022

@author: Lenovo
"""

from torch import nn
import torch.nn.functional as F

class MyAwesomeModel(nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()
        self.fc1 = nn.Linear(num_input, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_output)
        
    def forward(self, x):
        # make sure input tensor is flattened
        #x = x.view(x.shape[0], -1)
        
        x = x.float()
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.log_softmax(self.fc4(x), dim=1)
        
        return x