class BaseModel(object):
    def __init__(self, cfg, params):
        self.cfg = cfg
        self.n = float(params['population'])
        self.S = params['S']
        self.I = params['I']
        self.R = params['R']
        self.D = params['D']
        