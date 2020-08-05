import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

sns.set_style("whitegrid", {'grid.linestyle': '--',})

def RMSE(y_pred, y_true):
    return round(np.mean(np.sqrt((y_pred-y_true)**2)), 2)

def plot_pair_lines(y_pred, y_true, pred_label, true_label, title, X=None,
    pred_color='darkorange', true_color='darkgreen', save_dir=None):
    
    # plt.figure(figsize=(8,8))
    if X is None:
        X = np.arange(len(y_pred))
    
    max_val = abs(max(max(y_pred), max(y_true)))*1.25
    interval_width = max_val/6
    y_range = np.arange(-max_val, max_val+interval_width, interval_width)

    sns.lineplot(X, y_pred, marker='o', color=pred_color, label=pred_label)
    pR = sns.lineplot(X, y_true, marker='o', color=true_color, label=true_label)
    pR.set(yticks=y_range)
    plt.title(title)
    plt.legend()
    plt.show()

    if save_dir is not None:
        plt.savefig(save_dir)
        plt.cfg()
    
def visualize_error_rate(val_params, pred_results, x_axis=None, save_dir=None):
    # plt.figure(figsize=(8,8))
    I_pred = pred_results['I_pred']
    R_pred = pred_results['R_pred']
    I_val = val_params['I']
    R_val = val_params['R']


    I_error_rate = (I_pred - I_val)/I_val
    R_error_rate = (R_pred - R_val)/R_val 

    plot_pair_lines(
        y_pred = I_error_rate,
        y_true = R_error_rate,
        pred_label='Prediction error of infected persons',
        true_label='Prediction error of recovered persons',
        title=f"Error rate, $RMSE_I$: {RMSE(I_pred, I_val)}, $RMSE_R$: {RMSE(R_pred, R_val)}",
        X=x_axis, save_dir=save_dir
    )

def modify_zero_value(list_vals):
    res = np.array(list_vals)
    min_val = min(np.array([i for i in list_vals if i != 0.0]))
    for i in range(len(list_vals)):
        if list_vals[i] == 0.0:
            res[i] = min_val*0.5
    return res

def visualize_R0(val_params, pred_params, x_axis=None, save_dir=None):
    gamma_pred = modify_zero_value(pred_params['gamma_pred'])
    gamma_true = modify_zero_value(val_params['gamma'])
    Ro_pred = pred_params['beta_pred']/gamma_pred
    Ro = val_params['beta']/gamma_true

    plot_pair_lines(Ro_pred, Ro, pred_label="$\hat{R}_0 (t)$", true_label="$R_0 (t)$", 
        title="Comparison of original and predicted $R_0$", X=x_axis[:-1], save_dir=save_dir)


def visualize_params(val_params, pred_params, param, x_axis=None, save_dir=None):
    if param not in ['beta', 'gamma', 'delta']:
        print(f"invalid parameter: {param}")
        return

    param_pred = pred_params[f'{param}_pred']
    param_true = val_params[param]

    plot_pair_lines(param_pred, param_true, pred_label=f"Predicted {param}(t)", true_label = f"{param}(t)", 
    title = f"Comparison of original and predicted {param}", X=x_axis[:-1], save_dir=save_dir)


def visualize_all_result(val_params, pred_params, x_axis=None, save_dir=None):
    visualize_error_rate(val_params, pred_params, x_axis, save_dir)
    visualize_R0(val_params, pred_params, x_axis, save_dir)

    visualize_params(val_params, pred_params, 'beta', x_axis, save_dir)
    visualize_params(val_params, pred_params, 'gamma', x_axis, save_dir)
    if 'D_pred' in pred_params.keys():
        visualize_params(val_params, pred_params, 'delta', x_axis, save_dir)
        

