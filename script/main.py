import os
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from time_sir import TimeSIR
from time_sird import TimeSIRD
from data_utils import train_test_split, prepare_data, remove_year
from default_config import get_default_config, population, SEIR_STATE
from basic_sir import BasicSIR
from basic_sird import BasicSIRD
from visualize_utils import visualize_all_result, visualize_result, plot_single_set, \
                            visualize_error_rate, visualize_basic_result, visualize_R0_from_all

cfg = get_default_config()

def load_data(test_size=0.1, country='US', state = "Texas"):
    data_dir = os.path.join(cfg.cwd, cfg.data.root)
    data_dir = os.path.join(data_dir, country)
    file_dir = os.path.join(data_dir, state+'.csv')

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
    print(f"--- test time-SIR ---")
    train_params, val_params, raw_params = get_data_params(train_df, val_df, raw_df, False, state, country)

    model = TimeSIR(cfg, train_params)
    model.train()
    res = model.predict(val_params)
    

    filename = f"{country}/{state}_timeSIR"
    save_dir = os.path.join(cfg.data.save_path, filename)  
    visualize_result(raw_params, res, x_axis=raw_df['Day'].values, log=False, save_dir=save_dir)
    visualize_R0_from_all(raw_params, res, log=False, save_dir=save_dir, x_axis=raw_df['Day'].values[:-1])
    visualize_all_result(val_params, res, x_axis=val_df['Day'].values, save_dir=save_dir)

def test_time_SIRD(train_df, val_df, raw_df, state = "Texas", country='US'):
    print(f"--- test time-SIRD ---")
    train_params, val_params, raw_params = get_data_params(train_df, val_df, raw_df, True, state, country)

    model = TimeSIRD(cfg, train_params)
    model.train()
    res = model.predict(val_params)
    
    
    filename = f"{country}/{state}_timeSIRD"
    save_dir = os.path.join(cfg.data.save_path, filename)  
    visualize_result(raw_params, res, is_have_death = True, log = False, save_dir=save_dir, x_axis=raw_df['Day'].values)
    visualize_R0_from_all(raw_params, res, log=False, save_dir=save_dir, x_axis=raw_df['Day'].values[:-1])
    visualize_all_result(val_params, res, x_axis=val_df['Day'].values, save_dir=save_dir)

def test_basic_SIR(raw_df, attribute2fix: str, state='Texas', country='US'):
    '''
        attribute2fix: ['I', 'R']
    '''
    print(f"--- Basic SIR ---")
    raw_params = prepare_data(
        raw_df['Confirmed'].values,
        raw_df['Deaths'].values,
        raw_df['Recovered'].values,
        population[country][state]
    )

    filename = f"{country}/{state}_basicSIR"
    save_dir = os.path.join(cfg.data.save_path, filename) 

    basic_sir = BasicSIR(cfg, raw_params)
    res = basic_sir.fit_single_attribute(attribute=attribute2fix, visualize=False)
    print("finish curve fitting")
    val_params = {'I': basic_sir.I, 'R': basic_sir.R}
    x_axis=raw_df['Day'].values

    visualize_basic_result(val_params, res, x_axis, save_dir, attribute2fix)

def test_basic_SIRD(raw_df, attribute2fix: str, state='Texas', country='US'):
    '''
        attribute2fix: ['I', 'R']
    '''
    print(f"--- Basic SIRD ---")
    raw_params = prepare_data(
        raw_df['Confirmed'].values,
        raw_df['Deaths'].values,
        raw_df['Recovered'].values,
        population[country][state],
        is_have_death=True
    )

    filename = f"{country}/{state}_basicSIRD"
    save_dir = os.path.join(cfg.data.save_path, filename) 

    basic_sir = BasicSIRD(cfg, raw_params)
    res = basic_sir.fit_single_attribute(attribute=attribute2fix, visualize=False)
    print("finish curve fitting")
    val_params = {'I': basic_sir.I, 'R': basic_sir.R, 'D': basic_sir.D}
    x_axis=raw_df['Day'].values

    plot_single_set(raw_params, x_axis, save_dir)
    visualize_basic_result(val_params, res, x_axis, save_dir, attribute2fix)


def test_data(country, state):
    train_df, val_df, raw_df = load_data(country=country, state = state)

    raw_params = prepare_data(
        raw_df['Confirmed'].values,
        raw_df['Deaths'].values,
        raw_df['Recovered'].values,
        population[country][state]
    )

    print(f'test on {state}')
    # plot_single_set(raw_params)
    # test_time_SIR(train_df, val_df, raw_df, state, country)
    # test_time_SIRD(train_df, val_df, raw_df, state, country)

    for att in ['I', 'R']:
        test_basic_SIR(raw_df, att, state, country)
        test_basic_SIRD(raw_df, att, state, country)


if __name__ == "__main__":
    list_countries = list(population.keys())

    for country in list_countries:
        # if country != 'US': 
        #     continue
        for state in population[country].keys():
            # if state in SEIR_STATE or state == 'Texas':
            #     continue
            test_data(country, state)
 