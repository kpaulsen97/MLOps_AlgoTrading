# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 15:47:24 2022

@author: Lenovo
"""

import requests
import datetime
import pandas  as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame 
import numpy as np


# note that CLIENT_ID refers to 'personal use script' and SECRET_TOKEN to 'token'
auth = requests.auth.HTTPBasicAuth('ImXWi9rH09Q7P_VoeAyUTw', 'J3tLzP4r3KIH2bjKtaagaD00AYCgRA')

# here we pass our login method (password), username, and password
data = {'grant_type': 'password',
        'username': 'Creepy_Ferret_5915',
        'password': 'X8<kepf97'}

# setup our header info, which gives reddit a brief description of our app
headers = {'User-Agent': 'MyBot/0.0.1'}

# send our request for an OAuth token
res = requests.post('https://www.reddit.com/api/v1/access_token',
                    auth=auth, data=data, headers=headers)

# convert response to JSON and pull access_token value
TOKEN = res.json()['access_token']

# add authorization to our headers dictionary
headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}

# while the token is valid (~2 hours) we just add headers=headers to our requests
requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)

res = requests.get("https://oauth.reddit.com/r/python/hot",
                   headers=headers)

def df_from_response(res):
    df = pd.DataFrame()  # initialize dataframe

    # loop through each post retrieved from GET request
    for post in res.json()['data']['children']:
    # append relevant data to dataframe
        df = df.append({
            'date': datetime.datetime.utcfromtimestamp(post['data']['created']),
            'title': post['data']['title'],
            'selftext': post['data']['selftext'],
            'upvote_ratio': post['data']['upvote_ratio'],
            'ups': post['data']['ups']
        }, ignore_index=True)
    return df


data = pd.DataFrame()
params = {'limit': 100}

# loop through 10 times (returning 1K posts)
for i in range(3):
    # make request
    res = requests.get("https://oauth.reddit.com/r/apple/new",
                       headers=headers,
                       params=params)

    # get dataframe from response
    new_df = df_from_response(res)
    # take the final row (oldest entry)
    #row = new_df.iloc[len(new_df)-1]
    # create fullname
    #fullname = row['kind'] + '_' + row['id']
    fullname = res.json()['data']['children'][len(new_df)-1]['data']['name']
    # add/update fullname in params
    params['after'] = fullname
    
    # append new_df to data
    data = data.append(new_df, ignore_index=True)
    
    
tokenizer = AutoTokenizer.from_pretrained("poom-sci/bert-base-uncased-multi-emotion")

model = AutoModelForSequenceClassification.from_pretrained("poom-sci/bert-base-uncased-multi-emotion")

emotions = model.config.id2label.values()

def from_string_to_emotion(phrase):
    inputs = tokenizer(phrase, return_tensors="pt")
    
    with torch.no_grad():
        logits = model(**inputs).logits
        
        
    return pd.Series(logits.flatten())

title_emotions = ["title_"+emo for emo in emotions]

data[title_emotions] = data.title.apply(lambda x: from_string_to_emotion(x))

data.date = data.date.apply(lambda x: datetime.datetime(x.year, x.month, x.day))

idx = data.date < datetime.datetime(2022, 8, 8)
data = data.loc[idx, :]

#Remove weekends
idx = data.date.apply(lambda x: x.weekday() < 5)
data = data.loc[idx, :]
max_date = data.date.max()
min_date = data.date.min()

# API Info for fetching data, portfolio, etc. from Alpaca
BASE_URL = "https://paper-api.alpaca.markets"
ALPACA_API_KEY = "PKVVNQSH2VC2KPY14FNW"
ALPACA_SECRET_KEY = "DjHaHkcuyNjy3jV4Ww3rMCaXHFudNZbeyxrFYWAC"

# Instantiate REST API Connection
api = tradeapi.REST(key_id=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, 
                    base_url=BASE_URL, api_version='v2')

# Fetch Account
account = api.get_account()


data_aapl = api.get_bars("AAPL", TimeFrame.Day, str(min_date)[:10], str(max_date+datetime.timedelta(days=1))[:10], adjustment='raw').df

target = data_aapl.vwap

target_comp = target.copy()
target_comp[-1] = np.nan
target_comp[:-1] = target.copy()[1:]

target_final = target_comp > target

target_final = target_final.apply(lambda x: int(x))[:-1]

target_df = pd.DataFrame(target_final.values, columns=['target'])
target_df['date'] = target_final.keys()
target_df.date = target_df.date.apply(lambda x: datetime.datetime(x.year,x.month,x.day))

complete_df = pd.merge(data, target_df, how='inner', on='date')

X = torch.from_numpy(complete_df.drop(["selftext","date","title"],1).values)
y = complete_df['target']
X_train = X[:170,:]
X_test = X[170:,:]
y_train = y[:170]
y_test = y[170:]

processed = {'x_train':X_train, 'x_test':X_test, 'y_train':y_train,'y_test':y_test}

torch.save(processed, 'C:\\Users\\Lenovo\\Desktop\\University\\Machine Learning Operations\\Final_Project\\AlgoRepo\\data\\processed\\processed.pt')