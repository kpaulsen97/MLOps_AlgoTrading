# -*- coding: utf-8 -*-
"""
Created on Sun Aug 14 16:22:56 2022

@author: Lenovo
"""

from torch.utils.data import DataLoader, Dataset
import torch


class DDP(Dataset):
    def __init__(self, path_to_folder: str, key) -> None:
        # TODO: fill out with what you need
        self.data = torch.load(path_to_folder)[key]
    
    def __getitem__(self, index: int) -> torch.Tensor:
        # TODO: fill out
        return self.data[index]

dataset = DDP("C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\data"+"/processed/processed.pt", "x_train")



datal = DataLoader(
    dataset,
    batch_size=8,
    num_workers=4  # this is the number of threds we want to parallize workload over
)


