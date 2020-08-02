import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt

from time_sir import TimeSIR
from data_utils import train_test_split, prepare_data, visualize_result, plot_single_set
from default_config import get_default_config
from scipy import integrate, optimize
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import mean_squared_error


STATE = "Texas"

def main(cfg):
    data_dir = os.path.join(cfg.cwd, cfg.data.root)
    file_dir = os.path.join(data_dir, STATE+'.csv')
    
    raw_df = pd.read_csv(file_dir)
    
    train_df, val_df = train_test_split(raw_df, 0.1)
    cfg.model.predict_day = len(val_df)

    # train_params = prepare_data(train_df['Active'].values, 
    #                                 train_df['Confirmed'].values,
    #                                 train_df['Deaths'].values,
    #                                 train_df['Recovered'].values,
    #                                 cfg.population.texas)

    # val_parms = prepare_data(val_df['Active'].values, 
    #                                 val_df['Confirmed'].values,
    #                                 val_df['Deaths'].values,
    #                                 val_df['Recovered'].values,
    #                                 cfg.population.texas)
    
    raw_params = prepare_data(raw_df['Active'].values, 
                                    raw_df['Confirmed'].values,
                                    raw_df['Deaths'].values,
                                    raw_df['Recovered'].values,
                                    cfg.population.texas)

    # model = TimeSIR(cfg, train_params)
    # model.train()
    # res = model.predict(val_parms)
    # visualize_result(raw_params, res)

    # plot_single_set(raw_params)


    # -----------------------------------------------------------
    # BASIC SIR:
    N = raw_params['population']
    I = raw_params['I']
    S = raw_params['S']
    R = raw_params['R']
    
    I0 = I[0]
    S0 = S[0]
    R0 = R[0]
    x = np.arange(len(I))

    def sir_model(f, x, beta, gamma):
        s0, i0, r0 = f
        s = -beta*s0*i0/N 
        r = gamma*i0 
        i = -(s + r)
        return s, i, r
    
    def fit_odeint(x, beta, gamma):
        return integrate.odeint(sir_model, (S0, I0, R0), x, 
                                args=(beta, gamma))[:, 2]

    popt, pcov = optimize.curve_fit(fit_odeint, x, R)
    
    beta, gamma = popt
    print(f"beta: {beta}, gamma: {gamma}")
    
    # I_pred = fit_odeint(x, beta, gamma)
    R_pred = fit_odeint(x, beta, gamma)
    #S_pred = fit_odeint(x, beta, gamma, "suspected")
    # R_pred = integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 2]
    I_pred = integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, 1]

    plt.plot(x, I, label='$I(t)$', color='orange')
    plt.plot(x, I_pred, label='$\hat{I}(t)$', color='blue')

    plt.plot(x, R, label='$R(t)$', color='limegreen')
    plt.plot(x, R_pred, label='$\hat{R}(t)$', color='red')

    plt.legend()
    plt.show()

    


if __name__ == "__main__":
    cfg = get_default_config()
    main(cfg)