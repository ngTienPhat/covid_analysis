import numpy as np 

def RMSE(y_pred, y_true):
    return round(np.mean(np.sqrt((y_pred-y_true)**2)), 2)

def modify_zero_value(list_vals):
    res = np.array(list_vals)
    min_val = min(np.array([i for i in list_vals if i != 0.0]))
    for i in range(len(list_vals)):
        if list_vals[i] == 0.0:
            res[i] = min_val*0.5
    return res

def R0(beta, gamma):
    gamma = modify_zero_value(gamma)
    return beta/gamma

def error_rate(y_pred, y_true):
    y_deno = y_true
    for i in range(len(y_deno)):
        if y_deno[i] == 0:
            y_deno[i] = 1
    return (y_pred-y_true)/y_deno
