import os
import pandas as pd

from model import TimeSIR
from data_utils import train_test_split, prepare_data, visualize_result
from default_config import get_default_config

STATE = "Texas"

def main(cfg):
    data_dir = os.path.join(cfg.cwd, cfg.data.root)
    file_dir = os.path.join(data_dir, STATE+'.csv')
    
    raw_df = pd.read_csv(file_dir)
    
    train_df, val_df = train_test_split(raw_df, 0.1)
    cfg.model.predict_day = len(val_df)

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

    model = TimeSIR(cfg, **train_params)
    model.train()
    res = model.predict(val_parms)
    visualize_result(raw_params, res)

if __name__ == "__main__":
    cfg = get_default_config()
    main(cfg)