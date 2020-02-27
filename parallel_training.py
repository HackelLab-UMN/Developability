import load_datasets_for_model as loadmodel
import plotting_functions
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from functools import partial
from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'




def format_input(train_formatted,test_formatted,model_name,assays):
    # input is part A + part B
    #Part B is one-hot encoded variables of either the assays or the cell types 
    if '2' in model_name:
        b_train=np.asarray(train_formatted['Assay_ID'].values.tolist()).squeeze()
        b_test=np.asarray(test_formatted['Assay_ID'].values.tolist()).squeeze()
    else:
        b_train=train_formatted[['IQ_binary','SH_binary']].to_numpy()
        b_test=test_formatted[['IQ_binary','SH_binary']].to_numpy()

    if 'control' in model_name: #To use as "zero-rule" control that only guesses via type of cell predicting or type of assay predicting
        model_train_x=b_train
        model_test_x=b_test
    else:
        #Part A is either the sequence (ordinal or one_hot) or the assay scores
        if 'emb' in model_name: #keep emb first cause of seqemb_fnn_small
            a_train=np.asarray(train_formatted['Ordinal'].values.tolist())
            a_test=np.asarray(test_formatted['Ordinal'].values.tolist())
        elif 'seq' in model_name:
            a_train=np.asarray(train_formatted['One_Hot'].values.tolist())
            a_test=np.asarray(test_formatted['One_Hot'].values.tolist())
        elif 'assay' in model_name:
            a_train=train_formatted[assays].values.tolist()
            a_test=test_formatted[assays].values.tolist()

        model_train_x=np.concatenate((a_train,b_train),axis=1)
        model_test_x=np.concatenate((a_test,b_test),axis=1)

    return model_train_x,model_test_x

def train_model(space,model_name,save_model,model_train_x,model_train_y):
    if 'nn' in model_name:
        import tensorflow as tf
        import tf_models_local
        tf.keras.backend.clear_session() #helps reset tf_session
        input_size=np.shape(model_train_x)[1]
        if 'fnn_small' in model_name:
            model=tf_models_local.fnn_small(space,input_size,model_name)
        elif 'fnn' in model_name:
            model=tf_models_local.fnn(space,input_size,model_name)
        elif 'cnn' in model_name:
            model=tf_models_local.cnn(space,input_size,model_name)
        elif 'rnn' in model_name:
            model=tf_models_local.rnn(space,input_size,model_name)
        model.compile(optimizer='adam',loss=tf.keras.losses.MeanSquaredError())
        model.fit(model_train_x,model_train_y,epochs=int(10**space['epochs']),batch_size=int(space['batch_size']),verbose=0)
        if save_model: 
            #I'm having trouble saving models with attn layers****
            # json_config = model.to_json()
            # with open('./final_models/'+model_name+'_'+str(save_model)+'.json', 'w') as json_file:
            #     json_file.write(json_config)
            # Save weights to disk
            # model.save_weights('./final_models/'+model_name+'_'+str(save_model)+'.h5')
            model.save('./final_models/'+model_name+'_'+str(save_model)+'.h5')
    else:
        if 'ridge' in model_name:
            from sklearn.linear_model import Ridge
            model=Ridge(alpha=10**space['alpha'])
        elif 'forest' in model_name:
            from sklearn.ensemble import RandomForestRegressor
            model=RandomForestRegressor(n_estimators=int(space['n_estimators']),max_depth=int(space['max_depth']),max_features=space['max_features'],n_jobs=3)
        elif 'svm' in model_name:
            from sklearn.svm import SVR
            model=SVR(gamma=10**space['gamma'],C=10**space['c'])
        model.fit(model_train_x,model_train_y)
        if save_model:
            from joblib import dump
            dump(model,'./final_models/'+model_name+'_'+str(save_model)+'.joblib')
    return model


def parallel_part(space,model_name,assays,save_model,cv_sets):
    #for a given set of data, return the loss of predicting the test set
    train=cv_sets[0]
    test=cv_sets[1]

    #format dataset by dividing up datapoints of unique proteins
    if '2' in model_name:
        #explode datapoints by assay scores (for seq w/ >1 assay score)
        train_formatted=loadmodel.load_seq_to_assay_data(train) 
        test_formatted=loadmodel.load_seq_to_assay_data(test)
        y_name='Score'
    else:
        #explode datapoints by yields (for seq w/ IQ+SH yields)
        train_formatted=loadmodel.load_x_to_yield_data(train) 
        test_formatted=loadmodel.load_x_to_yield_data(test)
        y_name='Yield'
    
    #format input/outputs 
    model_train_y=train_formatted[y_name].to_numpy()
    model_test_y=test_formatted[y_name].to_numpy()
    model_train_x,model_test_x=format_input(train_formatted,test_formatted,model_name,assays)

    #Calculate experimental variance to compare to model error
    train_var=np.mean(np.square(train_formatted[y_name+'_std'].to_numpy()))
    test_var=np.mean(np.square(test_formatted[y_name+'_std'].to_numpy()))

    #train model 
    model=train_model(space,model_name,save_model,model_train_x,model_train_y)

    #use trained model to determine training error and test error
    train_predictions=model.predict(model_train_x)
    train_err=mse(train_predictions,model_train_y)
    test_predictions=model.predict(model_test_x).squeeze()
    test_err=mse(test_predictions,model_test_y)

    del model
    return train_var,train_err,test_var,test_err,model_test_y,test_predictions
