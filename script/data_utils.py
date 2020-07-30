import numpy as np
import pandas as pd

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

def prepare_data(arr_active, arr_confirm, arr_death, arr_recover, population):
    pops = np.array(len(arr_active)*[population], dtype=np.float)

    I = arr_confirm - arr_recover - arr_death
    R = arr_recover + arr_death
    S = pops - I - R 

    gamma = (R[1:] - R[:-1])/I[:-1]
    beta = pops[:-1]*(I[1:]-I[:-1] + R[1:]-R[:-1]) / (I[:-1]*S[:-1])


    params = {
        'I': I, 'R': R, 'S': S, 'gamma': gamma, 'beta': beta, 'population' : population
    }
    return params