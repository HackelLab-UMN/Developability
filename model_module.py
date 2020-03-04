import numpy as np
import pandas as pd
import pickle
from hyperopt import Trials,fmin,tpe,STATUS_OK
from sklearn.model_selection import RepeatedKFold
import load_format_data
from model_architectures import get_model

class model:
    '''The model class will cross-validate the training set to determine hyperparameters
    then use the set hyperparameters to evaluate against a test set and save predictions''' 

    def __init__(self, model_in, model_out, model_architecture, sample_fraction):
        self.sample_fraction=sample_fraction
        self.model_name=model_in+'_'+model_out+'_'+model_architecture+'_'+str(sample_fraction)
        self.trials_file='./trials/'+self.model_name+'.pkl'

        self.build_architecture(model_architecture) #ridge,forest,svm,nn(emb,)
        self.load_hyp()

    def parent_warning(self):
        print('im in parent class')

    def build_architecture(self, model_architecture):
        'load architecture class which sets hyp space'
        self._model=get_model(model_architecture)

    def evaluate_model_cv(self):
        'train the repeated kfold dataset. Caclulate average across splits of one dataset, then average across repeats of dataset'
        for i in self.data_pairs:
            train_x,train_y=self.format_modelIO(i[0])
            test_x,test_y=self.format_modelIO(i[1])

    def evaluate_model_test(self):
        'train the reapeated training data. Calculate average loss on test set to average out model randomness'
        pass

    # def predict_model(self):
    #     'useing trained model, predict test set'
    #     self.parent_warning()

    # def save_predictions(self):
    #     'updata dataframe with new column of predicted values from model'
    #     self.parent_warning()

    def load_hyp(self):
        'load hyperopt trials'
        try:  # try to load an already saved trials object
            self.tpe_trials = pickle.load(open(self.trials_file, "rb"))
            self.best_trial = load_format_data(self.tpe_trials)
        except:
            self.tpe_trials = Trials()

    def save_hyp(self):
        'save hyperopt trials, refresh best trial'
        with open(self.trials_file, "wb") as f:
            pickle.dump(self.tpe_trials, f)
        self.best_trial = load_format_data(self.tpe_trials)

    # def save_model(self):
    #     'save the trained model'
    #     self.parent_warning()

    # def apply_model(self):
    #     'using best hyp, train models using all training data and predict independent test set'
    #     self.parent_warning()

    def format_modelIO(self,df):
        'based upon model architecture and catagorical variables create the numpy input (x) and output (y) for the model'
        df_local=self.get_output_and_explode(df) #set y, do output firest to explode cat variables
        df_local=self.get_input_seq(df_local) #set xa (OH seq, Ord seq, assay, control)
        df_local=load_format_data.mix_with_cat_var(df_local) #mix xa with cat variables


        x=df_local['x']
        y=df_local['y']
        print(x,y)
        return x,y

    def make_cv_dataset(self):
        'create list of subtraining/validation by repeated cv of training data'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction)
        kf=RepeatedKFold(n_splits=self.num_cv_splits,n_repeats=self.num_cv_splits)
        train,validate=[],[]
        for train_index, test_index in kf.split(np.zeros(len(local_df))):
            train.append(local_df.iloc[train_index])
            validate.append(local_df.iloc[test_index])
        self.data_pairs=zip(train,validate)

    def make_test_dataset(self,num_test_repeats=10):
        'create list of full training set/test set for repeated model performance evaluation'
        local_df=load_format_data.sub_sample(self.training_df,self.sample_fraction)
        train,test=[],[]
        for i in range(num_test_repeats):
            train.append(local_df)
            test.append(self.testing_df)
        self.data_pairs=zip(train,test)

    def set_model_state(self,cv):
        'create list of paired dataframes and determine how to calculate loss based upon cross-validaiton or applying to test set'
        if cv:
            self.evaluate_model=self.evaluate_model_cv
            self.make_cv_dataset()
        else:
            self.evaluate_model=self.evaluate_model_test
            self.make_test_dataset()

    def hyperopt_obj(self,space):
        'for a given hyperparameter set, build model arch, evaluate model, return validation loss'
        self.set_model_state(cv=True)
        self._model.set_model(space)
        self.evaluate_model()

        return {'loss': 0, 'status': STATUS_OK ,'hyperparam':space}

    def cross_validate_model(self):
        'use hpyeropt to determine hyperparameters for self.tpe_trials'
        if len(self.tpe_trials)<self.num_hyp_trials:
            tpe_best=fmin(fn=self.hyperopt_obj,space=self._model.parameter_space,algo=tpe.suggest,trials=self.tpe_trials,max_evals=len(self.tpe_trials)+3)
            self.save_hyp()
        else:
            print('Already done with cross-validation')
