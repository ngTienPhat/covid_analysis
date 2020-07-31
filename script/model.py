import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

from data_utils import data_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

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

class TimeSIR(object):
    def __init__(self, cfg, **kwargs):
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

        self.cfg = cfg
        self.n = kwargs['population']
        self.S = kwargs['S']
        self.I = kwargs['I']
        self.R = kwargs['R']
        self.beta = kwargs['beta']
        self.gamma = kwargs['gamma']

        self.x_beta, self.y_beta = data_split(self.beta, cfg.model.orders_beta, cfg.model.start_beta)
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

    def predict(self, val_params):
        cfg = self.cfg
        S_pred = [self.S[-1]]
        R_pred = [self.R[-1]]
        I_pred = [self.I[-1]]
        pred_beta = np.array(self.beta[-cfg.model.orders_beta:])
        pred_gamma = np.array(self.gamma[-cfg.model.orders_gamma:])

        cnt_day = 0
        turning_point = 0
        
        while (I_pred[-1] >= 0) and (cnt_day < cfg.model.predict_day):
            if pred_beta[-1] > pred_gamma[-1]:
                turning_point += 1

            next_beta = self.linear_models['beta'].predict(pred_beta[-cfg.model.orders_beta:].reshape((1,-1)))[0]
            next_gamma = self.linear_models['gamma'].predict(pred_gamma[-cfg.model.orders_gamma:].reshape((1,-1)))[0]
    
            next_beta = max(next_beta, 0)
            next_gamma = max(next_gamma, 0)

            next_S = ((-next_beta * S_pred[-1] * I_pred[-1])/self.n) + S_pred[-1]
            next_I = (1+next_beta-next_gamma)*I_pred[-1]
            next_R = R_pred[-1] + next_gamma*I_pred[-1]

            S_pred.append(next_S)
            I_pred.append(next_I)
            R_pred.append(next_R)

            np.insert(pred_beta, -1, next_beta)
            np.insert(pred_gamma, -1, next_gamma)

            cnt_day += 1

        result = {
            "S_pred": S_pred[1:],
            "I_pred": I_pred[1:],
            "R_pred": R_pred[1:],
            "turning_point": turning_point
        }

        return result

        # plt.plot(range(len(I_pred)-1), I_pred[1:], '*-', label=r'$\hat{I}(t)$', color='darkorange')
        # plt.plot(range(len(I_pred)-1), R_pred[1:], '*-', label=r'$\hat{R}(t)$', color='limegreen')
        # plt.plot(range(len(val_params['I'])), val_params['I'], '--', label=r'$I(t)$', color='chocolate')
        # plt.plot(range(len(val_params['I'])), val_params['R'], '--', label=r'$R(t)$', color='darkgreen')
        # plt.xlabel('Day')
        # plt.ylabel('Person')
        # plt.title('Time evolution of the time-dependent SIR model.')
        # plt.legend()
        # plt.show()


