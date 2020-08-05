import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

sns.set_style("whitegrid", {'grid.linestyle': '--',})

def RMSE(y_pred, y_true):
    return round(np.mean(np.sqrt((y_pred-y_true)**2)), 2)

def visualize_error_rate(val_params, pred_results, x_axis=None):
    plt.figure(figsize=(8,8))
    I_pred = pred_results['I_pred']
    R_pred = pred_results['R_pred']
    I_val = val_params['I']
    R_val = val_params['R']


    I_error_rate = (I_pred - I_val)/I_val
    R_error_rate = (R_pred - R_val)/R_val 

    X = np.arange(len(I_pred))
    if x_axis != None:
        X = x_axis

    sns.lineplot(X, I_error_rate, marker='o', color='darkorange',
                label='Prediction error of infected persons')
    pR = sns.lineplot(X, R_error_rate, marker='o', color='darkgreen',
                label='Prediction error of recovered persons')
    
    max_distance = max(
        max(abs(I_error_rate)), 
        max(abs(R_error_rate))
    )
    interval_width = max_distance/10
    y_range = np.arange(-max_distance, max_distance+interval_width, interval_width)

    pR.set(yticks=y_range)
    # pI.set(yticks=[i for i in np.arange(min(min(I_error_rate), min(R_error_rate))-0.1, 
    #                                     max(max(I_error_rate), max(R_error_rate))+0.1, 
    #                                     0.05)])
    
    plt.title(f"Error rate, $RMSE_I$: {RMSE(I_pred, I_val)}, $RMSE_R$: {RMSE(R_pred, R_val)}")
    plt.legend()
    plt.show()
