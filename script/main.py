import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy import integrate, optimize
from sklearn.metrics import mean_squared_error

from time_sir import TimeSIR
from data_utils import train_test_split, prepare_data, visualize_result, plot_single_set
from default_config import get_default_config
from basic_sir import BasicSIR

cfg = get_default_config()

def load_data(test_size=0.1):
    STATE = "Texas"
    data_dir = os.path.join(cfg.cwd, cfg.data.root)
    file_dir = os.path.join(data_dir, STATE+'.csv')

    raw_df = pd.read_csv(file_dir)
    train_df, val_df = train_test_split(raw_df, test_size)
    cfg.model.predict_day = len(val_df)

    return train_df, val_df, raw_df


def test_time_SIR(train_df, val_df, raw_df):
    train_params = prepare_data(train_df['Active'].values, 
                                    train_df['Confirmed'].values,
                                    train_df['Deaths'].values,
                                    train_df['Recovered'].values,
                                    cfg.population.texas)

    val_parms = prepare_data(val_df['Active'].values, 
                                    val_df['Confirmed'].values,
                                    val_df['Deaths'].values,
                                    val_df['Recovered'].values,
                                    cfg.population.texas)
    
    raw_params = prepare_data(raw_df['Active'].values, 
                                    raw_df['Confirmed'].values,
                                    raw_df['Deaths'].values,
                                    raw_df['Recovered'].values,
                                    cfg.population.texas)

    model = TimeSIR(cfg, train_params)
    model.train()
    res = model.predict(val_parms)
    visualize_result(raw_params, res)

    plot_single_set(raw_params)


def test_basic_SIR(raw_df, attribute2fix: str):
    '''
        attribute2fix: ['I', 'R']
    '''

    raw_params = prepare_data(raw_df['Active'].values, 
                                    raw_df['Confirmed'].values,
                                    raw_df['Deaths'].values,
                                    raw_df['Recovered'].values,
                                    cfg.population.texas)
    
    basic_sir = BasicSIR(cfg, raw_params)
    basic_sir.fit_single_attribute(attribute='R', visualize=True)


if __name__ == "__main__":
    train_df, val_df, raw_df = load_data()
    
    ## A. test time-SIR 
    # test_time_SIR(train_df, val_df, raw_df)

    ## B. test basic SIR with curve fit:
    # test_basic_SIR('I')
    test_basic_SIR(raw_df, 'R')