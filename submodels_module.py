import numpy as np
import pickle
from functools import partial
from model_module import model
import load_format_data
import plot_model


class seq_to_x_model():
    'sets get_input_seq to ordinal or onehot sequence based upon model_architecture'
    def __init__(self, model_architecture):
        if 'emb' in model_architecture:
            self.get_input_seq=load_format_data.get_ordinal
        else:
            self.get_input_seq=load_format_data.get_onehot

class assay_to_x_model():
    'sets get_input_seq to assay scores of assays'
    def __init__(self, assays):
        self.get_input_seq=partial(load_format_data.get_assays,assays)

class control_to_x_model():
    'sets get_input_seq to nothing, not sure if needed'
    def __init__(self):
        self.get_input_seq=load_format_data.get_control 

class x_to_yield_model(model):
    'sets model output to yield'
    def __init__(self, model_in, model_architecture, sample_fraction):
        super().__init__(model_in, 'yield', model_architecture, sample_fraction)
        self.get_output_and_explode=load_format_data.explode_yield
        self.plot_type=plot_model.x_to_yield_plot
        self.training_df=load_format_data.load_df('assay_to_dot_training_data')
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data') 
        self.num_cv_splits=10
        self.num_cv_repeats=10
        self.num_hyp_trials=50

    def save_predictions(self):
        'saves model predictions for the large dataset'
        df=load_format_data.load_df('seq_to_assay_train_all10') #will have to adjust if missing datapoints
        OH_matrix=np.eye(2)
        matrix_col=['IQ_Average_bc','SH_Average_bc']
        x_a=self.get_input_seq(df)
        for z in range(3): #no of models
            self.load_model(z)
            for i in range(2):
                cat_var=[]
                for j in x_a:
                    cat_var.append(OH_matrix[i].tolist())
                x=load_format_data.mix_with_cat_var(x_a,cat_var)
                df_prediction=self._model.model.predict(x).squeeze().tolist()
                col_name=matrix_col[i]
                df.loc[:,col_name]=df_prediction
            df.to_pickle('./datasets/predicted/seq_to_assay_train_all10_'+self.model_name+'_'+str(z)+'.pkl')



class x_to_assay_model(model):
    'sets to assay_model'
    def __init__(self, model_in, assays, model_architecture, sample_fraction):
        super().__init__(model_in, 'assay'+str(assays), model_architecture, sample_fraction)
        self.get_output_and_explode=partial(load_format_data.explode_assays,assays)
        self.plot_type=plot_model.x_to_assay_plot
        self.training_df=load_format_data.load_df('seq_to_assay_train_all10') #could adjust in future for sequences with predictive assays
        self.testing_df=load_format_data.load_df('assay_to_dot_training_data')
        self.num_cv_splits=3
        self.num_cv_repeats=3
        self.num_hyp_trials=100


class assay_to_yield_model(x_to_yield_model, assay_to_x_model):
    'assay to yield, provide which assays, limit test set to useable subset'
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('assays'+str(assays), model_architecture, sample_fraction)
        assay_to_x_model.__init__(self,assays)
        #Limit test set for data that has all assay scores used in model
        sort_names=[]
        for i in assays:
            sort_names.append('Sort'+str(i)+'_mean_score')
        dataset=self.testing_df
        dataset=dataset[~dataset[sort_names].isna().any(axis=1)] 
        self.testing_df=dataset

class seq_to_yield_model(x_to_yield_model, seq_to_x_model):
    'seq to yield'
    def __init__(self, model_architecture, sample_fraction):
        super().__init__('seq', model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)

class seq_to_pred_yield_model(x_to_yield_model,seq_to_x_model):
    'sequence to yield model using predicted yields from assay scores'
    def __init__(self, pred_yield_model_prop, seq_to_pred_yield_prop):
        super().__init__('seq_pred',seq_to_pred_yield_prop[0],seq_to_pred_yield_prop[1])
        seq_to_x_model.__init__(self,seq_to_pred_yield_prop[0])
        pred_yield_model_name='assays'+str(pred_yield_model_prop[0])+'_yield_'+pred_yield_model_prop[1]+'_'+str(pred_yield_model_prop[2])+'_'+str(pred_yield_model_prop[3])
        self.training_df=load_format_data.load_df('predicted/seq_to_assay_train_all10_'+pred_yield_model_name)
        self.num_cv_splits=3
        self.num_cv_repeats=3
        self.num_hyp_trials=50


class seq_to_assay_model(x_to_assay_model, seq_to_x_model):
    'seq to assay, provide assays'
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('seq',assays, model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)

class control_to_assay_model(x_to_assay_model, control_to_x_model):
    'predict assay scores based upon average of assay score of training set'
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('control',assays, model_architecture, sample_fraction)
        control_to_x_model.__init__(self)

class control_to_yield_model(x_to_yield_model, control_to_x_model):
    'predict assay scores based upon average of assay score of training set'
    def __init__(self, model_architecture, sample_fraction):
        super().__init__('control', model_architecture, sample_fraction)
        control_to_x_model.__init__(self)



