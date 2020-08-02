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

def visualize_result(all_params, pred_result):
    I_pred = pred_result['I_pred']
    R_pred = pred_result['R_pred']
    n_pred = len(I_pred)

    I_all = all_params['I']
    R_all = all_params['R']
    n = len(I_all)


    # Plot all value:
    plt.plot(range(n), I_all, '--', label=r'$I(t)$', color='darkorange')
    plt.plot(range(n), R_all, '--', label=r'$R(t)$', color='limegreen')

    plt.plot(range(n-n_pred, n), I_pred, '--+', label=r'$\hat{I}(t)$', color='red')
    plt.plot(range(n-n_pred, n), R_pred, '--+', label=r'$\hat{R}(t)$', color='blue')


   
    plt.xlabel('Day')
    plt.ylabel('Person')
    plt.title('Time evolution of the time-dependent SIR model.')
    plt.legend()
    plt.show()


def plot_single_set(set_params):
    I = set_params['I']
    R = set_params['R']

    S = np.log(np.array(set_params['S']))
    # print(S[:10])
    n = len(I)

    plt.plot(range(n), I, '--', label=r'$I(t)$', color='darkorange')
    plt.plot(range(n), R, '--', label=r'$R(t)$', color='limegreen')
    # plt.plot(range(n), S, '--', label=r'$S(t)$', color='blue')

    plt.xlabel('Day')
    plt.ylabel('Person')
    plt.legend()
    plt.show()