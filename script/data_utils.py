import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def train_test_split(df, test_size=0.2):
    total_day = len(df) 
    test_size = int(total_day*test_size)
    train_size = total_day - test_size

    return df.copy().iloc[:train_size, :], df.copy().iloc[train_size:, :]

def data_split(x, orders, start):
    x_train = np.zeros((len(x)-start-orders, orders))
    y_train = x[start+orders:]

    for i in range(len(x)-start-orders):
        x_train[i] = x[start+i:start+i+orders]
    
    return x_train, y_train

list_delims = ['-', '/']
def remove_year(full_date):
    day_month = None
    for delim in list_delims:
        if delim in full_date:
            day_month = full_date.split(delim)[:-1]
            res = '/'.join(day_month)
            break
    return res

def prepare_data(arr_confirm, arr_death, arr_recover, population, is_have_death=False):
    pops = np.array(len(arr_confirm)*[population], dtype=np.float)

    I, R, D, S, gamma, beta, delta = 0, 0, 0, 0, 0, 0, 0
  
    if is_have_death:
        I = arr_confirm - arr_recover - arr_death
        R = arr_recover
        D = arr_death
        S = pops - I - R - D
        
        gamma = (R[1:] - R[:-1])/I[:-1]
        beta = population*(S[:-1]-S[1:]) / (I[:-1]*S[:-1])
        delta = (D[1:] - D[:-1])/I[:-1]
    else:
        I = arr_confirm - arr_recover - arr_death
        R = arr_recover + arr_death
        D = 0
        S = pops - I - R - D
        gamma = (R[1:] - R[:-1])/I[:-1]
        beta = population*(S[:-1]-S[1:]) / (I[:-1]*S[:-1])
        delta = 0

    params = {
        'I': I, 'R': R, 'S': S, 'gamma': gamma, 'beta': beta, 'population' : population, 'D': D, 'delta': delta
    }
    return params
