from model_module import model
from functools import partial
import load_format_data


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
        self.training_df=load_format_data.load_df('assay_to_dot_training_data')
        self.testing_df=load_format_data.load_df('seq_to_dot_test_data') 
        self.num_cv_splits=10
        self.num_hyp_trials=50

class x_to_assay_model(model):
    'sets to assay_model'
    def __init__(self, model_in, assays, model_architecture, sample_fraction):
        super().__init__(model_in, 'assay'+str(assays), model_architecture, sample_fraction)
        self.get_output_and_explode=partial(load_format_data.explode_assays,assays)
        self.training_df=load_format_data.load_df('seq_to_assay_train_all10') #could adjust in future for sequences with predictive assays
        self.testing_df=load_format_data.load_df('assay_to_dot_training_data')
        self.num_cv_splits=3
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

class seq_to_assay_model(x_to_assay_model, seq_to_x_model):
    'seq to assay, provide assays'
    def __init__(self, assays, model_architecture, sample_fraction):
        super().__init__('seq',assays, model_architecture, sample_fraction)
        seq_to_x_model.__init__(self,model_architecture)


