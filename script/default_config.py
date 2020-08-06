from yacs.config import CfgNode as CN

SEIR_STATE = ['Tianjin', 'Chongqing', 'Jilin', 'Zhejiang']
population = {
    'US': {
        'Texas': 29087070,
        'New York': 19491339,
        'Pennsylvania': 12813969,
        'Ohio': 11718568,
        'Michigan': 10020472,
    },
    'China':{
        'Beijing': 11716620,
        'Tianjin': 11090314,
        'Chongqing': 7457600,
        'Jilin': 1881977,
        'Shanghai': 22315474,
        'Zhejiang': 632552,
        # 'Hubei': 58520000
    },
}

def get_default_config():
    cfg = CN() 

    # working directory
    cfg.cwd = "/Users/tienphat/Documents/HCMUS/Statistic_Application/project/covid_analysis"

    cfg.data = CN()
    
    # data
    cfg.data.root = "dataset/csse_combine_state"    
    cfg.data.save_path = "RESULT" ## For example: We run timeSIR data of "US/Texas", results will be saved under "RESULT/US/Texas_timeSIR"
    # cfg.population = CN()


    # -----------------------Time_SIR--------------------------------
    cfg.model=CN()
    cfg.model.timeSIR_grid = True #Use grid search to train Ridge Regression or not
    cfg.model.timeSIRD_grid = True#Use grid search to train Ridge Regression or not

    # Learning params:
    cfg.model.orders_beta = 2
    cfg.model.orders_gamma = 2
    cfg.model.orders_delta = 2

    cfg.model.start_beta = 10
    cfg.model.start_gamma = 10
    cfg.model.start_delta = 10
    
    cfg.model.predict_day = 20 #max day to predict, W in the paper

    return cfg
