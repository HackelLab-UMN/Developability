import pandas as pd
import numpy as np
import random

def load_df(file_name):
    'set name where datasets are found'
    return pd.read_pickle('./datasets/'+file_name+'.pkl')

def sub_sample(df,sample_fraction):
    'randomly sample dataframe'
    sub=random.sample(range(len(df)),int(len(df)*sample_fraction))
    return df.iloc[sub]

def get_best_trial(tpe_trials):
    'sort trials by loss, return best trial'
    sorted_trials = sorted(tpe_trials.results, key=lambda x: x['loss'], reverse=False)
    return sorted_trials[0]


def explode_yield(df):
    '''seperate datapoints by IQ/SH yield
    df: seq  IQ_yield SH_yield
         1     10          20

    produces:
        seq  cat_var yield(y)
        1       [1,0]   10 
        1       [0,1]   20

    '''
    OH_matrix=np.eye(2)
    IQ_data=df[df['IQ_Average_bc'].notnull()]
    if not IQ_data.empty:
        IQ_data.loc[:,'Cat_Var']=[[OH_matrix[0]]]*len(IQ_data)
        IQ_data.loc[:,'y']=IQ_data['IQ_Average_bc']

    SH_data=df[df['SH_Average_bc'].notnull()]
    if not SH_data.empty:
        SH_data.loc[:,'Cat_Var']=[[OH_matrix[1]]]*len(SH_data)
        SH_data.loc[:,'y']=SH_data['SH_Average_bc']

    return IQ_data.append(SH_data,ignore_index=True,sort=False)

def expode_assays(assays,df):
    'seperate datapoints by assays to be predicted similar to explode_yield'
    pass

def mix_with_cat_var(df):
    'mix x_a and cat_varable into model input x'
    xa=df['x_a'].values.tolist()
    cat_var=df['Cat_Var'].values.tolist()

    #need to figure out how to merge the different datatypes
    x_out=[]
    for i in range(len(xa)):
        print(xa[i].shape)
        print(np.array(cat_var[i][0]).shape)
        x_blah=np.concatenate(xa[i],np.array(cat_var[i][0]))
        print(x_blah)

    df[:,'x']=x_out

    return df

def get_ordinal(df):
    'sets ordinal encoded sequence to x_a in df for use in embedding models'
    pass

def get_onehot(df):
    'sets one_hot encoded sequence to x_a in df'
    df.loc[:,'x_a']=df.loc[:,'One_Hot']
    return df

def get_assays(assays,df):
    'sets assay values (of assays) to x_a in df'
    pass

def get_control(df):
    'x_a should be null for a zero_rule model that guesses based upon average'
    pass
