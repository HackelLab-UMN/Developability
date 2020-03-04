import hyperopt_model
import pandas as pd
import model_training_functions
import numpy as np

def test_final(model_name,assays):
    
    if '2' in model_name:
        training_data=pd.read_pickle('./seq_to_assay_train_subset.pkl') #train seq-2-assay model
        # training_data=training_data.iloc[0:1000]
    else:
        training_data=pd.read_pickle('./assay_to_dot_training_data.pkl')

    tpe_trials,_=hyperopt_model.load_hyp_trials(model_name)

    sorted_trials = sorted(tpe_trials.results, key=lambda x: x['loss'], reverse=False)

    best_space=sorted_trials[0]['hyperparam']
    test_err,test_std,test_var=model_training_functions.evaluate_model(training_data,best_space,model_name,assays=assays,cv=False)
    return test_err,test_std,test_var #avg test loss, std test loss, test exp var

def cross_validate_best_model(model_name,assays,resolve):
    if '2' in model_name:
        training_data=pd.read_pickle('./seq_to_assay_train_subset.pkl') #train seq-2-assay model
        # training_data=training_data.iloc[0:1000]
    else:
        training_data=pd.read_pickle('./assay_to_dot_training_data.pkl')

    tpe_trials,_=hyperopt_model.load_hyp_trials(model_name)
    sorted_trials = sorted(tpe_trials.results, key=lambda x: x['loss'], reverse=False)

    best_space=sorted_trials[0]['hyperparam']
    if resolve:
        train_data=model_training_functions.evaluate_model(training_data,best_space,model_name,assays=assays,cv=1,save_fig=True)
    return best_space,sorted_trials[0]['loss'],sorted_trials[0]['loss_std']