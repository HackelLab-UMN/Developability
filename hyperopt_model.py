# Hyper parameter optimization 
# Process- Get training set, get hyperspace by model name, see if trials can be loaded, run new trials, update trials

from hyperopt import hp,tpe,Trials,fmin,trials_from_docs
import pickle
from functools import partial

def get_file_name(model_name):
    #save files in different folders so I can find them easier 
    if '2' in model_name or 'learned' in model_name:
        file_name='./CV_data2/hyperopt'+model_name+'.pkl'
    else:
        file_name='./CV_data/hyperopt'+model_name+'.pkl'
    return file_name

def load_hyp_trials(model_name):
    trials_step = 5  # how many additional trials to do after loading saved trials
    if '2' in model_name: #change hpt trials based upon predicting yield or assay scores
        max_trials=50
    else:
        max_trials=100

    file_name=get_file_name(model_name)
    
    try:  # try to load an already saved trials object, and increase the max
        tpe_trials = pickle.load(open(file_name, "rb"))
        new_trials = len(tpe_trials.trials) + trials_step # how many trials should be completed by end of fmin during this loop
        if new_trials>max_trials: #places a cap on hyp trials to do
            new_trials=max_trials
    except:  # create a new trials object and start searching
        tpe_trials = Trials()
        new_trials=trials_step
    return tpe_trials,new_trials


def update_hyp_trials(model_name,tpe_trials_cur):
    file_name=get_file_name(model_name)
    try:
        #update trials rather than save current incase I had mulitple running in parallel and instance finished during the training of another
        tpe_trials_old = pickle.load(open(file_name, "rb"))
        list_new=list(tpe_trials_old)
        list_new.extend(x for x in list(tpe_trials_cur) if x not in list_new)
        tpe_trials_updated=trials_from_docs(list_new)
    except:
        tpe_trials_updated=tpe_trials_cur
    with open(file_name, "wb") as f:
        pickle.dump(tpe_trials_updated, f)
   
    return tpe_trials_updated

def hyperopt_obj(dataset,model_name,assays,sample_size,space):
    from hyperopt import STATUS_OK
    import model_training_functions    
    if sample_size: #used to subsample database, feature that needs to be implemented better
            import random
            sub=random.sample(range(len(dataset)),sample_size)
            dataset=dataset.iloc[sub]

    if '2' in model_name:
        cv_data=model_training_functions.evaluate_model(dataset,space,model_name,assays,cv=3)
    else:
        cv_data=model_training_functions.evaluate_model(dataset,space,model_name,assays,cv=10)
    return {'loss': cv_data[3], 'status': STATUS_OK , 'train_loss':cv_data[1], 'test_var':cv_data[2],'loss_std':cv_data[4],'hyperparam':space}

def get_space(model_name):
    #choose hyperparameter ranges for each type of model 
    if 'ridge' in model_name:
        space={'alpha':hp.uniform('alpha',-5,5)}

    elif 'forest' in model_name:
        space={'n_estimators':hp.quniform('n_estimators', 1, 500,1),'max_depth':hp.quniform('max_depth',1,100,1),'max_features':hp.uniform('max_features',0,1)}

    elif 'svm' in model_name:
        space={'gamma':hp.uniform('gamma', -3, 3),'c':hp.uniform('c',-3,3)}

    elif 'fnn_small' in model_name:
        space={'epochs':hp.uniform('epochs', 0, 2),'batch_size':hp.quniform('batch_size',10,200,1),\
            'layers':hp.quniform('layers',1,5,1),'nodes':hp.quniform('nodes',1,100,1)}
        if 'emb' in model_name:
            space['emb_dim']=hp.quniform('emb_dim',1,20,1)

    elif 'nn' in model_name:
        space={'epochs':hp.uniform('epochs', 0, 3),'layers':hp.quniform('layers',1,5,1),'nodes':hp.quniform('nodes',1,100,1),\
                'dense_drop':hp.uniform('dense_drop',0.1,0.5)}
        if '2' in model_name:
            space['batch_size']=hp.quniform('batch_size',100,5000,1)
        else:
            space['batch_size']=hp.quniform('batch_size',10,200,1)

        if 'emb' in model_name:
            space['emb_dim']=hp.quniform('emb_dim',1,100,1)

        if 'cnn' in model_name:
            space['filters']=hp.quniform('filters',1,100,1)
            space['kernel_size']=hp.quniform('kernel_size',1,16,1)
            space['input_drop']=hp.uniform('input_drop',0.1,0.5)
            space['cov_drop']=hp.uniform('cov_drop',0.1,0.5)

        elif 'rnn' in model_name:
            space['units']=hp.quniform('units',1,100,1)
            space['input_drop']=hp.uniform('input_drop',0.1,0.5)
            space['recurrent_drop']=hp.uniform('recurrent_drop',0.1,0.5)

    return space

def hyp_train(model_name,assays,sample_size=None):
    import pandas as pd
    #import CV dataset
    if '2' in model_name: #"2" used to determine if predicting yield or assay
        training_data=pd.read_pickle('./seq_to_assay_train_subset.pkl') 
    else:
        training_data=pd.read_pickle('./assay_to_dot_training_data.pkl')
    #get hyperparameter space by model name
    space=get_space(model_name)
    #load old trials 
    tpe_trials,max_trials=load_hyp_trials(model_name)
    #wrap objective function for compatability with hyperopt learn
    hyperopt_obj_type=partial(hyperopt_obj,training_data,model_name,assays,sample_size)
    #fmin runs the object function by selecting the next hyperparameters based upon the space and previous trials 
    tpe_best=fmin(fn=hyperopt_obj_type,space=space,algo=tpe.suggest,trials=tpe_trials,max_evals=max_trials)
    #save trials for next loop
    tpe_trials=update_hyp_trials(model_name,tpe_trials)
    tpe_results = pd.DataFrame(list(tpe_trials.results))
    if '2' in model_name or 'learned' in model_name:
        tpe_results.to_csv('./CV_data2/'+model_name+'.csv')
    else:
        tpe_results.to_csv('./CV_data/'+model_name+'.csv')
