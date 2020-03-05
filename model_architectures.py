from hyperopt import hp
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

def get_model(model_architecture):
    'call this to set model._model based upon model_architecture'
    model_switcher= {
        'ridge': ridge_model(),
        'emb_ridge': ridge_model(),
        'forest': forest_model(),
        'svm': svm_model()
        }
    return model_switcher.get(model_architecture)

class ridge_model():
    def __init__(self):
        self.parameter_space={
        'alpha':hp.uniform('alpha', -5, 5)
        }

    def set_model(self,space):
        self.model=Ridge(alpha=10**space['alpha'])

class forest_model():
    def __init__(self):
        self.parameter_space={
        'n_estimators':hp.quniform('n_estimators', 1, 500, 1),
        'max_depth':hp.quniform('max_depth', 1, 100, 1),
        'max_features':hp.uniform('max_features', 0, 1)
        }
    def set_model(self,space):
        self.model=RandomForestRegressor(n_estimators=int(space['n_estimators']),max_depth=int(space['max_depth']),max_features=space['max_features'])

class svm_model():
    def __init__(self):
        self.parameter_space={
        'gamma':hp.uniform('gamma', -3, 3),
        'c':hp.uniform('c', -3, 3)
        }
    def set_model(self,space):
        self.model=SVR(gamma=10**space['gamma'],C=10**space['c'])




