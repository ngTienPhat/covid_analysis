from yacs.config import CfgNode as CN

population = {
    'US': {
        'Texas': 29087070,
        'California': 39747267,
        'New York': 19491339,
        'Pennsylvania': 12813969,
        'Illinois': 12700381,
        'Ohio': 11718568,
        'Georgia': 10627767,
        'Michigan': 10020472,
    },
    'China':{
        'Beijing': 11716620,
        'Tianjin': 11090314,
        'Chongqing': 7457600,
        'Jilin': 1881977,
        'Shanghai': 22315474,
        'Zhejiang': 632552,
    },
}

def get_default_config():
    cfg = CN() 

    # working directory
    cfg.cwd = "/Users/tienphat/Documents/HCMUS/Statistic_Application/project/covid_analysis"

    cfg.data = CN()
    
    # data
    cfg.data.root = "dataset/csse_combine_state"    

    # cfg.population = CN()



    # -----------------------Time_SIR--------------------------------
    cfg.model=CN()
    # Learning params:
    cfg.model.orders_beta = 2
    cfg.model.orders_gamma = 2
    cfg.model.orders_delta = 2

    cfg.model.start_beta = 10
    cfg.model.start_gamma = 10
    cfg.model.start_delta = 10
    
    cfg.model.predict_day = 20 #max day to predict, W in the paper

    return cfg
