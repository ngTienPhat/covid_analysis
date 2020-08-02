from default_model import BaseModel
from data_utils import data_split

class BasicSIR(BaseModel):
    def __init__(self, cfg, params):
        super(BasicSIR, self).__init__(cfg, params)

    self.beta_final = None 
    self.gamma_final = None 

    # def fit(self)


