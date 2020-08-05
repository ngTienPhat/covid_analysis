import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV


from default_model import BaseModel
from data_utils import data_split

def ridge(x, y):
    print('\nStart searching good parameters for the task...')
    parameters = {'alpha': np.arange(0, 0.100005, 0.000005).tolist(),
                  "tol": [1e-8],
                  'fit_intercept': [True, False],
                  'normalize': [True, False]}

    clf = GridSearchCV(Ridge(), parameters, n_jobs=-1, cv=5)
    clf.fit(x, y)

    print('\nResults for the parameters grid search:')
    print('Model:', clf.best_estimator_)
    print('Score:', clf.best_score_)

    return clf


class TimeSIRD(BaseModel):
    def __init__(self, cfg, params):
        super(TimeSIRD, self).__init__(cfg, params)
        self.linear_models = dict()
        self.linear_models['beta'] = Ridge(alpha=0.003765, 
                                copy_X=True, 
                                fit_intercept=False,
                                max_iter=None, 
                                normalize=True, 
                                random_state=None, 
                                solver='auto', 
                                tol=1e-08)

        self.linear_models['gamma'] = Ridge(alpha=0.003765, 
                                copy_X=True, 
                                fit_intercept=False,
                                max_iter=None, 
                                normalize=True, 
                                random_state=None, 
                                solver='auto', 
                                tol=1e-08)

        self.linear_models['delta'] = Ridge(alpha=0.003765, 
                                copy_X=True, 
                                fit_intercept=False,
                                max_iter=None, 
                                normalize=True, 
                                random_state=None, 
                                solver='auto', 
                                tol=1e-08)

        self.beta = params['beta']
        self.gamma = params['gamma']
        self.delta = params['delta']

        self.x_beta, self.y_beta = data_split(self.beta, cfg.model.orders_beta, cfg.model.start_beta)
        self.x_delta, self.y_delta = data_split(self.delta, cfg.model.orders_delta, cfg.model.start_delta)
        self.x_gamma, self.y_gamma = data_split(self.gamma, cfg.model.orders_gamma, cfg.model.start_gamma)

    def fit_linear(self, X, y, variable="gamma"):
        self.linear_models[variable].fit(X, y)
        # self.linear_models[variable] = ridge(X, y)

    def evaluate_linear(self, X_test, y_test, variable="gamma"):
        y_hat = self.linear_models[variable].predict(X_test)
        plt.plot(y_hat, label=f"predicted {variable}")
        plt.plot(y_test, label=f"true {variable}")
        plt.legend()

    def train(self):
        self.fit_linear(self.x_beta, self.y_beta, "beta") 
        print(f"finish training beta")

        self.fit_linear(self.x_gamma, self.y_gamma, "gamma")
        print(f"finish training gamma")

        self.fit_linear(self.x_gamma, self.y_gamma, "delta")
        print(f"finish training delta")

    def predict(self, val_params):
        cfg = self.cfg
        S_pred = [self.S[-1]]
        R_pred = [self.R[-1]]
        I_pred = [self.I[-1]]
        D_pred = [self.D[-1]]
        pred_beta = np.array(self.beta[-cfg.model.orders_beta:])
        pred_gamma = np.array(self.gamma[-cfg.model.orders_gamma:])
        pred_delta = np.array(self.delta[-cfg.model.orders_delta:])

        cnt_day = 0
        turning_point = 0
        
        while (I_pred[-1] >= 0) and (cnt_day < cfg.model.predict_day):
            if pred_beta[-1] > pred_gamma[-1]:
                turning_point += 1

            next_beta = self.linear_models['beta'].predict(pred_beta[-cfg.model.orders_beta:].reshape((1,-1)))[0]
            next_gamma = self.linear_models['gamma'].predict(pred_gamma[-cfg.model.orders_gamma:].reshape((1,-1)))[0]
            next_delta = self.linear_models['delta'].predict(pred_delta[-cfg.model.orders_delta:].reshape((1,-1)))[0]

            next_beta = max(next_beta, 0.001)
            next_gamma = max(next_gamma, 0.001)
            next_delta = max(next_delta, 0.001)

            next_S = ((-next_beta * S_pred[-1] * I_pred[-1])/self.n) + S_pred[-1]
            # next_I = (1+next_beta-next_gamma)*I_pred[-1]
            next_I = next_beta*S_pred[-1]*I_pred[-1]/self.n - (next_gamma + next_delta)*I_pred[-1] + I_pred[-1]
            next_R = R_pred[-1] + next_gamma*I_pred[-1]
            next_D = next_delta*I_pred[-1] + D_pred[-1]

            S_pred.append(next_S)
            I_pred.append(next_I)
            R_pred.append(next_R)
            D_pred.append(next_D)

            pred_beta = np.insert(pred_beta, len(pred_beta), next_beta)
            pred_gamma = np.insert(pred_gamma, len(pred_gamma), next_gamma)
            pred_delta = np.insert(pred_delta, len(pred_delta), next_delta)

            cnt_day += 1

        result = {
            "S_pred": S_pred[1:],
            "I_pred": I_pred[1:],
            "R_pred": R_pred[1:],
            "D_pred": D_pred[1:],
            "beta_pred": pred_beta[cfg.model.orders_beta:-1],
            "gamma_pred": pred_gamma[cfg.model.orders_gamma:-1],
            "delta_pred": pred_delta[cfg.model.orders_gamma:-1],
            "turning_point": turning_point
        }

        return result
       