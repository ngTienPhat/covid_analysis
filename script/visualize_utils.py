import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import os
import os.path as osp
from evaluation import RMSE, error_rate, R0

sns.set_style("whitegrid", {'grid.linestyle': '--',})

def mkdir_if_not_exist(filedir):
    if filedir is None:
        return
    if not os.path.exists(filedir):
        os.makedirs(filedir)
        print(f'created folder {filedir} to save result')

def get_xticks_ids(x_axis):
    n = len(x_axis)
    final_step = 4
    # for i in range(6, 2, -1):
    #     if (n-1)%i == 0:
    #         final_step=i

    X = x_axis
    Xi = range(n)
    raw_ids = np.ones(n)
    ids = list(range(0, n, final_step))
    if n-1 not in ids:
        ids.append(n-1)
    return ids

def plot_pair_lines(y_pred, y_true, pred_label, true_label, title, X=None,
    pred_color='darkorange', true_color='darkgreen', save_dir=None):
    
    # plt.figure(figsize=(8,8))
    if X is None:
        X = np.arange(len(y_pred))
    else:
        xticks_ids = get_xticks_ids(X)

    max_val = abs(max(max(y_pred), max(y_true)))*1.25
    interval_width = max_val/6
    y_range = np.arange(-max_val, max_val+interval_width, interval_width)

    sns.lineplot(X, y_pred, marker='o', color=pred_color, label=pred_label)
    pR = sns.lineplot(X, y_true, marker='o', color=true_color, label=true_label)
    pR.set(yticks=y_range)
    plt.xticks(xticks_ids, X)
    plt.title(title)
    plt.legend()

    if save_dir is not None:
        plt.savefig(save_dir)
        plt.clf()
    
    else:
        plt.show()
    
def visualize_error_rate(val_params, pred_results, x_axis=None, save_dir=None):
    # plt.figure(figsize=(8,8))
    I_pred = pred_results['I_pred']
    R_pred = pred_results['R_pred']
    I_val = val_params['I']
    R_val = val_params['R']

    I_error_rate = error_rate(I_pred, I_val)
    R_error_rate = error_rate(R_pred, R_val)

    if save_dir is not None:
        save_dir = osp.join(save_dir, "error_rate.jpg")

    plot_pair_lines(
        y_pred = I_error_rate,
        y_true = R_error_rate,
        pred_label='Prediction error of infected persons',
        true_label='Prediction error of recovered persons',
        title=f"Error rate, $RMSE_I$: {RMSE(I_pred, I_val)}, $RMSE_R$: {RMSE(R_pred, R_val)}",
        X=x_axis, save_dir = save_dir
    )

def visualize_R0(val_params, pred_params, x_axis=None, save_dir=None):
    Ro_pred = R0(pred_params['beta_pred'], pred_params['gamma_pred']) 
    Ro = R0(val_params['beta'], val_params['gamma'])

    if save_dir is not None:
        save_dir = osp.join(save_dir, "R0.jpg")

    plot_pair_lines(Ro_pred, Ro, pred_label="$\hat{R}_0 (t)$", 
        true_label="$R_0 (t)$", 
        title="Comparison of original and predicted $R_0$", 
        X=x_axis[:-1], 
        save_dir=save_dir
    )

def visualize_params(val_params, pred_params, param, x_axis=None, save_dir=None):
    if param not in ['beta', 'gamma', 'delta']:
        print(f"invalid parameter: {param}")
        return

    param_pred = pred_params[f'{param}_pred']
    param_true = val_params[param]

    if save_dir is not None:
        save_dir = osp.join(save_dir, f"{param}.jpg")

    plot_pair_lines(param_pred, param_true, pred_label=f"Predicted {param}(t)", true_label = f"True {param}(t)", 
    title = f"Comparison of original and predicted {param}", X=x_axis[:-1], save_dir=save_dir)

def visualize_all_result(val_params, pred_params, x_axis=None, save_dir=None):
    mkdir_if_not_exist(save_dir)

    visualize_error_rate(val_params, pred_params, x_axis, save_dir)
    visualize_R0(val_params, pred_params, x_axis, save_dir)

    visualize_params(val_params, pred_params, 'beta', x_axis, save_dir)
    visualize_params(val_params, pred_params, 'gamma', x_axis, save_dir)
    if 'D_pred' in pred_params.keys():
        visualize_params(val_params, pred_params, 'delta', x_axis, save_dir)

def visualize_R0_from_all(all_params, pred_result, log = False, save_dir=None, x_axis=None):
    plt.figure(figsize=(25,8))
    beta_raw = all_params['beta']
    gamma_raw = all_params['gamma']
    beta_pred = pred_result['beta_pred']
    gamma_pred = pred_result['gamma_pred']

    R0_raw = R0(beta_raw, gamma_raw)
    R0_pred = R0(beta_pred, gamma_pred)

    n_pred = len(beta_pred)
    n = len(beta_raw)
    X = np.arange(n) 
    if x_axis is not None:
        xticks_ids = get_xticks_ids(x_axis)
        X = x_axis

    # Plot all value:
    plt.plot(X, [1]*n, '--', label=r'threshold $R_0 (t)$ < 1', color='lightgreen')
    plt.plot(X, R0_raw, '--*', label=r'$R_0 (t)$', color='darkorange')
    plt.plot(X[range(n-n_pred, n)], R0_pred, '--o', label=r'$\hat{R_0}(t)$', color='darkblue')
    plt.xticks(xticks_ids, X)
    
    plt.xlabel('Date')
    plt.ylabel('Reproduction number $R_0 (t)$')
    plt.legend()
    if save_dir is not None:
        mkdir_if_not_exist(save_dir)
        if log:
            figname = 'R0_prediction_log.jpg'
        else:
            figname = 'R0_prediction.jpg'
        plt.savefig(osp.join(save_dir, figname))
        plt.clf()
    else:
        plt.show()

def visualize_result(all_params, pred_result, is_have_death = False, log = False, save_dir=None, x_axis=None):
    '''
        Visualize the prediction for I(t) and R(t) in the future based on information gained from train set
        If save_dir is not None, the output figure is saved with the name "trend_prediction.jpg". Otherwise, plt.show()
    '''
    plt.figure(figsize=(25,8))
    I_pred = pred_result['I_pred']
    R_pred = pred_result['R_pred']
    if is_have_death:
        D_pred = pred_result['D_pred']
    n_pred = len(I_pred)

    I_all = all_params['I']
    R_all = all_params['R']
    D_all = all_params['D']
    n = len(I_all)

    X = np.arange(n) 
    if x_axis is not None:
        xticks_ids = get_xticks_ids(x_axis)
        X = x_axis

    if log:
      D_pred = np.log(D_pred)
      I_pred = np.log(I_pred)
      R_pred = np.log(R_pred)
      
      I_all = np.log(I_all)
      R_all = np.log(R_all)
      D_all = np.log(D_all)
    
    # Plot all value:
    plt.plot(X, I_all, '--', label=r'$I(t)$', color='darkorange')
    plt.plot(X, R_all, '--', label=r'$R(t)$', color='limegreen')

    plt.plot(X[range(n-n_pred, n)], I_pred, '--+', label=r'$\hat{I}(t)$', color='red')
    plt.plot(X[range(n-n_pred, n)], R_pred, '--+', label=r'$\hat{R}(t)$', color='blue')
    
    if is_have_death:
        plt.plot(X[range(n)], D_all, '--', label=r'$D(t)$', color='black')
        plt.plot(X[range(n-n_pred, n)], D_pred, '--+', label=r'$\hat{D}(t)$', color='purple')

    plt.xticks(xticks_ids, X)
    plt.xlabel('Day')
    plt.ylabel('Person')
    plt.title('Time evolution of the time-dependent SIR model.')
    plt.legend()
    if save_dir is not None:
        mkdir_if_not_exist(save_dir)
        if log:
            figname = 'trend_prediction_log.jpg'
        else:
            figname = 'trend_prediction.jpg'
        plt.savefig(osp.join(save_dir, figname))
        plt.clf()
    else:
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

def visualize_basic_result(val_params, pred_params, x_axis=None, save_dir=None):
    mkdir_if_not_exist(save_dir)
    plt.figure(figsize=(25,8))

    for param in val_params.keys():
        y_true = val_params[param]
        y_pred = pred_params[param+'_pred']
        
        if save_dir is not None:
            save_file = osp.join(save_dir, f"{param}.jpg")
        plot_pair_lines(
            y_pred,
            y_true,
            pred_label= f'Predicted {param}',
            true_label= f'True {param}',
            save_dir=save_file,
            title = f"Comparison of original and predicted {param}",
            X = x_axis
        )
    
    visualize_error_rate(val_params, pred_params, x_axis, save_dir)