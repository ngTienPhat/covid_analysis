from yacs.config import CfgNode as CN

def get_default_config():
    cfg = CN() 

    # working directory
    cfg.cwd = "/Users/tienphat/Documents/HCMUS/Statistic_Application/project/covid_analysis"

    cfg.data = CN()
    
    # data
    cfg.data.root = "dataset/csse_combine_state"    

    cfg.population = CN()

    # -----------------------POPULATION--------------------------------
    # US
    cfg.population.texas = 29087070
    cfg.population.california = 39747267
    cfg.population.florida = 21646155
    cfg.population.new_york = 19491339
    cfg.population.pennsylvania = 12813969
    cfg.population.illinois = 12700381
    cfg.population.ohio = 11718568
    cfg.population.georgia = 10627767
    cfg.population.michigan = 10020472

    # CHINA


    # -----------------------Time_SIR--------------------------------
    cfg.model=CN()
    # Learning params:
    cfg.model.orders_beta = 2
    cfg.model.orders_gamma = 2

    cfg.model.start_beta = 10
    cfg.model.start_gamma = 10
    
    cfg.model.predict_day = 20 #max day to predict, W in the paper

    return cfg
