# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 14:38:43 2022

@author: Lenovo
"""

text = open('AlgoRepo/requirements.txt').readlines()

text = [line[:-1] for line in text]

text = [line.split(' ')[0] for line in text]

text = [line.split('=')[0] for line in text]

with open('requirements.txt', 'w') as f:
    for line in text:
        f.write(f"{line}\n")