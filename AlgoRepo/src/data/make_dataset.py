# -*- coding: utf-8 -*-
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame 

import torch

import numpy as np


# API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "PKVVNQSH2VC2KPY14FNW"
ALPACA_SECRET_KEY = "DjHaHkcuyNjy3jV4Ww3rMCaXHFudNZbeyxrFYWAC"

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, 
                    base_url=BASE_URL, api_version='v2')

# Fetch Account
account = api.get_account()

#Take data 
data = api.get_bars("AAPL", TimeFrame.Hour, "2020-06-08", "2022-06-08", adjustment='raw').df



#store in torch
data_np = torch.from_numpy(data.to_numpy()[:8000,0].reshape(-1,20))

target = [0]
for i in range(1,len(data_np.flatten())):
    target.append(int(data_np.flatten()[i]>data_np.flatten()[i-1]))
    
targets = torch.from_numpy(np.array(target[20::20])).long()  

#x_train = mnist_trainset.data[:1000].view(-1, 784).float()
x_train = data_np[:300,:].float()
#targets_train = mnist_trainset.targets[:1000]
targets_train = targets[:300]


#x_valid = mnist_trainset.data[1000:1500].view(-1, 784).float()
x_valid = data_np[300:-1,:].float()
#targets_valid = mnist_trainset.targets[1000:1500]
targets_valid = targets[300:]


#x_test = mnist_testset.data[:500].view(-1, 784).float()
x_test = data_np[:100,:].float()
#targets_test = mnist_testset.targets[:500]
targets_test = targets[:100]

processed = {'x_train':x_train, 'x_valid':x_valid, 'x_test':x_test, 'targets_train':targets_train, 'targets_valid':targets_valid,'targets_test':targets_test}

torch.save(processed, 'C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\data\\processed\\processed.pt')



