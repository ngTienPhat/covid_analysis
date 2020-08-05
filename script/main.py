import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from scipy import integrate, optimize
from sklearn.metrics import mean_squared_error

from time_sir import TimeSIR
from time_sird import TimeSIRD
from data_utils import train_test_split, prepare_data, visualize_result, plot_single_set, remove_year
from default_config import get_default_config, population
from basic_sir import BasicSIR
from visualize_utils import visualize_all_result

cfg = get_default_config()

def load_data(test_size=0.1, country='US', STATE = "Texas"):
    data_dir = os.path.join(cfg.cwd, cfg.data.root)
    data_dir = os.path.join(data_dir, country)
    file_dir = os.path.join(data_dir, STATE+'.csv')

    raw_df = pd.read_csv(file_dir)
    raw_df['Day'] = raw_df['Day'].apply(remove_year)

    train_df, val_df = train_test_split(raw_df, test_size)
    cfg.model.predict_day = len(val_df)
    
    return train_df, val_df, raw_df


def get_data_params(train_df, val_df, raw_df, is_have_death=False, state="Texas", country='US'):
    train_params = prepare_data(train_df['Confirmed'].values,
                                train_df['Deaths'].values,
                                train_df['Recovered'].values,
                                population[country][state], is_have_death)

    val_parms = prepare_data(val_df['Confirmed'].values,
                            val_df['Deaths'].values,
                            val_df['Recovered'].values,
                            population[country][state], is_have_death)

    raw_params = prepare_data(raw_df['Confirmed'].values,
                            raw_df['Deaths'].values,
                            raw_df['Recovered'].values,
                            population[country][state], is_have_death)
    return train_params, val_parms, raw_params


def test_time_SIR(train_df, val_df, raw_df, state = "Texas", country='US'):
    train_params, val_params, raw_params = get_data_params(train_df, val_df, raw_df, False, state, country)

    model = TimeSIR(cfg, train_params)
    model.train()
    res = model.predict(val_params)
    visualize_result(raw_params, res, x_axis=None, log=True)
    visualize_all_result(val_params, res, x_axis=val_df['Day'].values)



def test_time_SIRD(train_df, val_df, raw_df, STATE = "Texas", country='US'):
    train_params, val_params, raw_params = get_data_params(train_df, val_df, raw_df, True, STATE, country)

    model = TimeSIRD(cfg, train_params)
    model.train()
    res = model.predict(val_params)
    visualize_result(raw_params, res, is_have_death = True, log = True, is_print = False, output_path = 'timeSIRD_'+ STATE +'.png')
    visualize_all_result(val_params, res, x_axis=val_df['Day'].values)


def test_basic_SIR(raw_df, attribute2fix: str, state='Texas', country='US'):
    '''
        attribute2fix: ['I', 'R']
    '''

    raw_params = prepare_data(
        raw_df['Confirmed'].values,
        raw_df['Deaths'].values,
        raw_df['Recovered'].values,
        population[country][state]
    )

    basic_sir = BasicSIR(cfg, raw_params)
    basic_sir.fit_single_attribute(attribute='R', visualize=True)


if __name__ == "__main__":

    # train_df, val_df, raw_df = load_data(STATE = "Texas")
    # test_time_SIR(train_df, val_df, raw_df)

    list_countries = list(population.keys())
    country = 'US'
    state = 'Texas'
    train_df, val_df, raw_df = load_data(country=country, STATE = state)
    
    raw_params = prepare_data(
        raw_df['Confirmed'].values,
        raw_df['Deaths'].values,
        raw_df['Recovered'].values,
        population[country][state]
    )
    
    print(f'test on {state}')
    plot_single_set(raw_params)
    # test_time_SIR(train_df, val_df, raw_df, state, country)
    test_time_SIRD(train_df, val_df, raw_df, state, country)
    # test_basic_SIR(raw_df, 'R')




    ## A. test time-SIR 
    # test_time_SIR(train_df, val_df, raw_df, state, country)

    # for state in population[country].keys():
    #     train_df, val_df, raw_df = load_data(country=country, STATE = state)
        
    #     print(f'test on {state}')

    #     ## A. test time-SIR 
    #     test_time_SIR(train_df, val_df, raw_df, state, country)
        

    #     ## B. test basic SIR with curve fit:
    #     # test_basic_SIR('I')
    #     # test_basic_SIR(raw_df, 'R')

    #     ## C. test time-SIRD:
    #     # test_time_SIRD(train_df, val_df, raw_df, STATE = state)
