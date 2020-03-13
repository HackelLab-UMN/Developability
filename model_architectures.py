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
        'fnn': fnn(),
        'emb_fnn_flat': emb_fnn_flat(),
        'emb_fnn_maxpool': emb_fnn_maxpool(),
        'emb_rnn': emb_rnn(),
        'emb_cnn': emb_cnn()
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

class nn():
    def __init__(self):
        self.parameter_space={
        'epochs':hp.uniform('epochs', 0, 3),
        'batch_size':hp.uniform('batch_size',0.1,1),
        'dense_layers':hp.quniform('dense_layers',1,5,1),
        'dense_nodes_per_layer':hp.quniform('dense_nodes_per_layer',1,100,1),
        'dense_drop':hp.uniform('dense_drop',0.1,0.5)
        }
    
    def set_init_model(self,space,**kwargs):
        tf.keras.backend.clear_session()
        self.space=space
        input_shape=kwargs['xa_len']+kwargs['cat_var_len']
        self.xa_len=kwargs['xa_len']

        self.inputs=tf.keras.Input(shape=(input_shape,))

    def dense_layers(self):
        layers=int(self.space['dense_layers'])
        nodes=int(self.space['dense_nodes_per_layer'])
        dense_dropout=self.space['dense_drop']
        dense,drop=[[]]*layers,[[]]*layers

        drop[0]=tf.keras.layers.Dropout(rate=dense_dropout)(self.recombined)
        
        ###following used for hidden layers
        dense[0]=tf.keras.layers.Dense(nodes,activation='relu')(drop[0])
        for i in range(1,layers):
            drop[i]=tf.keras.layers.Dropout(rate=dense_dropout)(dense[i-1])
            dense[i]=tf.keras.layers.Dense(nodes,activation='relu')(drop[i])
        
        ###final output uses last dropout layer
        self.outputs=tf.keras.layers.Dense(1,activation='linear')(drop[-1])  


    def set_end_model(self):
        self.model=tf.keras.Model(inputs=self.inputs,outputs=self.outputs)
        self.model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())

    def fit(self,x,y):
        self.epochs=int(10**self.space['epochs'])
        self.batch_size=int(len(x)*self.space['batch_size'])        
        self.model.fit(x,y,epochs=self.epochs,batch_size=self.batch_size,verbose=0)
        
class fnn(nn):
    def __init__(self):
        super().__init__()
        pass

    def set_model(self,space,**kwargs):
        self.set_init_model(space,**kwargs)
        self.recombined=self.inputs
        self.dense_layers()
        self.set_end_model()

    def get_seq_embeding_layer_model(self):
        print('this is just the regular OH encoding')
        return None

class emb_nn(nn):
    def __init__(self):
        super().__init__()
        self.parameter_space['AA_emb_dim']=hp.quniform('AA_emb_dim',1,20,1)

    def input_to_AA_emb(self):
        emb_dim=int(self.space['AA_emb_dim'])
        self.input_seq=tf.keras.layers.Lambda(lambda x: x[:,:self.xa_len])(self.inputs)
        self.AA_embed=tf.keras.layers.Embedding(21,emb_dim,input_length=16)(self.input_seq) #(batch size, seq len, embed size)

    def recombine_cat_var(self):
        self.input_cat_var=tf.keras.layers.Lambda(lambda x: x[:,self.xa_len:])(self.inputs)
        self.recombined=tf.keras.layers.concatenate([self.flat_seq,self.input_cat_var])

    def get_seq_embeding_layer_model(self):
        return tf.keras.Model(inputs=self.model.input,outputs=self.model.get_layer('seq_embedding').output)


class emb_fnn_maxpool(emb_nn):
    def __init__(self):
        super().__init__()
        pass

    def set_model(self,space,**kwargs):
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.flat_seq=tf.keras.layers.GlobalMaxPool1D(name='seq_embedding')(self.AA_embed) #pool across sequence len, end with (batch size, embed size)
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class emb_fnn_flat(emb_nn):
    def __init__(self):
        super().__init__()
        pass

    def set_model(self,space,**kwargs):
        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.flat_seq=tf.keras.layers.Flatten(name='seq_embedding')(self.AA_embed) #pool across sequence len, end with (batch size, embed size)
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class emb_rnn(emb_nn):
    def __init__(self):
        super().__init__()
        self.parameter_space['units']=hp.quniform('units',1,100,1)
        self.parameter_space['input_dropout']=hp.uniform('input_dropout',0.1,0.5)
        self.parameter_space['recurrent_dropout']=hp.uniform('recurrent_dropout',0.1,0.5)

    def set_model(self,space,**kwargs):
        units=int(space['units'])
        input_dropout=space['input_dropout']
        recurrent_dropout=space['recurrent_dropout']

        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.flat_seq=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(name='seq_embedding',units=units,recurrent_dropout=recurrent_dropout,dropout=input_dropout))(self.AA_embed)
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()

class emb_cnn(emb_nn):
    def __init__(self):
        super().__init__()
        self.parameter_space['filters']=hp.quniform('filters',1,100,1)
        self.parameter_space['kernel_size']=hp.quniform('kernel_size',1,16,1) #needs to be updated for different length sequences
        self.parameter_space['input_drop']=hp.uniform('input_drop',0.1,0.5)

    def set_model(self,space,**kwargs):
        filters=int(space['filters'])
        kernel_size=int(space['kernel_size'])
        input_drop=space['input_drop']

        self.set_init_model(space,**kwargs)
        self.input_to_AA_emb()
        self.input_drop=tf.keras.layers.Dropout(rate=input_drop)(self.AA_embed)
        self.cov=tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel_size,activation='relu')(self.input_drop)
        self.flat_seq=tf.keras.layers.GlobalMaxPool1D(name='seq_embedding')(self.cov)
        self.recombine_cat_var()
        self.dense_layers()
        self.set_end_model()



