# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 11:59:23 2022

@author: Lenovo
"""

from tests import _PATH_DATA, _PROJECT_ROOT
import torch
import sys

sys.path.append(
     _PROJECT_ROOT+"\\src\\models"
 )

from model import MyAwesomeModel

dataset = torch.load(_PATH_DATA+"/processed/processed.pt")
x_test = dataset['x_test']

num_input = len(x_test[0])

net = MyAwesomeModel(num_input, 2)


def test_model_dim():
    y = net(x_test) 
    assert y.shape == torch.Size([32,2])
    

