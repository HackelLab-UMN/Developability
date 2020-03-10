from hyperopt import hp
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf

def get_model(model_architecture):
    'call this to set model._model based upon model_architecture'
    model_switcher= {
        'ridge': ridge_model(),
        'forest': forest_model(),
        'svm': svm_model(),
        'small_fnn': small_fnn()
        }
    return model_switcher.get(model_architecture)

class ridge_model():
    def __init__(self):
        self.parameter_space={
        'alpha':hp.uniform('alpha', -5, 5)
        }
    def set_model(self,space,**kwargs):
        self.model=Ridge(alpha=10**space['alpha'])
    def fit(self,x,y):
        self.model.fit(x,y)

class forest_model():
    def __init__(self):
        self.parameter_space={
        'n_estimators':hp.quniform('n_estimators', 1, 500, 1),
        'max_depth':hp.quniform('max_depth', 1, 100, 1),
        'max_features':hp.uniform('max_features', 0, 1)
        }
    def set_model(self,space,**kwargs):
        self.model=RandomForestRegressor(n_estimators=int(space['n_estimators']),max_depth=int(space['max_depth']),max_features=space['max_features'])
    def fit(self,x,y):
        self.model.fit(x,y)

class svm_model():
    def __init__(self):
        self.parameter_space={
        'gamma':hp.uniform('gamma', -3, 3),
        'c':hp.uniform('c', -3, 3)
        }
    def set_model(self,space,**kwargs):
        self.model=SVR(gamma=10**space['gamma'],C=10**space['c'])
    def fit(self,x,y):
        self.model.fit(x,y)

class small_fnn():
    def __init__(self):
        self.parameter_space={
        'epochs':hp.uniform('epochs', 0, 2),
        'batch_size':hp.quniform('batch_size',10,200,1),
        'layers':hp.quniform('layers',1,5,1),
        'nodes':hp.quniform('nodes',1,100,1),
        'dense_drop':hp.uniform('dense_drop',0.1,0.5)
        }
    
    def set_model(self,space,**kwargs):
        tf.keras.backend.clear_session()
        layers=int(space['layers'])
        nodes=int(space['nodes'])
        dense_dropout=space['dense_drop']
        self.epochs=int(10**space['epochs'])
        self.batch_size=int(space['batch_size'])
        input_shape=kwargs['xa_len']+kwargs['cat_var_len']
        dense,drop=[[]]*layers,[[]]*layers

        inputs=tf.keras.Input(shape=(input_shape,))
        drop[0]=tf.keras.layers.Dropout(rate=dense_dropout)(inputs)
        
        ###following used for hidden layers
        dense[0]=tf.keras.layers.Dense(nodes,activation='relu')(drop[0])
        for i in range(1,layers):
            drop[i]=tf.keras.layers.Dropout(rate=dense_dropout)(dense[i-1])
            dense[i]=tf.keras.layers.Dense(nodes,activation='relu')(drop[i])
        
        ###final output uses last dropout layer
        outputs=tf.keras.layers.Dense(1,activation='linear')(drop[-1])  

        self.model=tf.keras.Model(inputs=inputs,outputs=outputs)
        self.model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())

    def fit(self,x,y):
        self.model.fit(x,y,epochs=self.epochs,batch_size=self.batch_size,verbose=0)
        



