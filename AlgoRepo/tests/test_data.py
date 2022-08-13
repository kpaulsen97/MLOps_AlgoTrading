# -*- coding: utf-8 -*-
"""
Created on Sat Aug 13 11:41:52 2022

@author: Lenovo
"""
from tests import _PATH_DATA
import torch

dataset = torch.load(_PATH_DATA+"/processed/processed.pt")
x_test = dataset['x_test']
x_train = dataset['x_train']

def test_N():
    assert x_test.shape[0]+x_train.shape[0] == 202
    
def test_sample_shape():
    assert x_train[0].shape == torch.Size([31])
