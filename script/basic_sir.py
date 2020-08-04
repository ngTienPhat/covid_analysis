import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate, optimize

from default_model import BaseModel
from data_utils import data_split

class BasicSIR(BaseModel):
    valid_params = ['I', 'R']

    def __init__(self, cfg, params):
        super(BasicSIR, self).__init__(cfg, params)

        self.final_beta = None 
        self.final_gamma = None 

    # def fit(self)

    def fit_single_attribute(self, attribute, visualize=True):
        if attribute not in self.valid_params:
            print (f"attribute must be {valid_params}")
            return
        
        att_idx = 1
        pred_y = None
        if attribute == 'I':
            att_idx = 1
            pred_y = self.I 
        elif attribute == 'R':
            att_idx = 2
            pred_y = self.R

        I0 = self.I[0]
        R0 = self.R[0]
        S0 = self.S[0]
        n = self.n
        X  = np.arange(len(self.I))

        
        def sir_model(f, x, beta, gamma):
            s0, i0, r0 = f
            s = -beta*s0*i0/n
            r = gamma*i0 
            i = -(s + r)
            return s, i, r
    
        def fit_odeint(x, beta, gamma):
            return integrate.odeint(sir_model, (S0, I0, R0), x, args=(beta, gamma))[:, att_idx]

        fitted_params, std_params = optimize.curve_fit(fit_odeint, X, pred_y)
        self.final_beta, self.final_gamma = fitted_params

        if visualize:
            I_pred = integrate.odeint(sir_model, (S0, I0, R0), X, args=(self.final_beta, self.final_gamma))[:, 1]
            R_pred = integrate.odeint(sir_model, (S0, I0, R0), X, args=(self.final_beta, self.final_gamma))[:, 2]
            plt.plot(X, self.I, label='$I(t)$', color='orange')
            plt.plot(X, I_pred, label='$\hat{I}(t)$', color='blue')

            plt.plot(X, self.R, label='$R(t)$', color='limegreen')
            plt.plot(X, R_pred, label='$\hat{R}(t)$', color='red')

            plt.legend()
            plt.show()
    
    # def predict(self):
    #     pass 

        
