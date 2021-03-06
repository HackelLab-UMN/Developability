import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit

def load_df(file_name):
    'set name where datasets are found'
    return pd.read_pickle('./datasets/'+file_name+'.pkl')

def sub_sample(df,sample_fraction):
    'randomly sample dataframe'
    sub=random.sample(range(len(df)),int(len(df)*sample_fraction))
    return df.iloc[sub]

def get_random_split(df):
    rs=ShuffleSplit(n_splits=1,train_size=300,random_state=42)
    for i,j in rs.split(list(range(len(df)))):
        train_i,test_i=i,j
    return df.iloc[train_i],df.iloc[test_i]

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
    cat_var=[]
    exploded_df=[]

    IQ_data=df[df['IQ_Average_bc'].notnull()]
    if not IQ_data.empty:
        for i in range(len(IQ_data)):
            cat_var.append(OH_matrix[0].tolist())
        IQ_data.loc[:,'y']=IQ_data['IQ_Average_bc']
        IQ_data.loc[:,'y_std']=IQ_data['IQ_Average_bc_std']
        exploded_df.append(IQ_data)

    SH_data=df[df['SH_Average_bc'].notnull()]
    if not SH_data.empty:
        for i in range(len(SH_data)):
            cat_var.append(OH_matrix[1].tolist())
        SH_data.loc[:,'y']=SH_data['SH_Average_bc']
        SH_data.loc[:,'y_std']=SH_data['SH_Average_bc_std']
        exploded_df.append(SH_data)

    exploded_df=pd.concat(exploded_df,ignore_index=True)
    y=exploded_df.loc[:,'y'].values.tolist()

    return exploded_df, cat_var, y

def explode_assays(assays,df):
    'seperate datapoints by assays to be predicted, similar to explode_yield'
    OH_matrix=np.eye(len(assays))
    OH_counter=0
    cat_var=[]
    exploded_df=[]

    for i in assays:
        assay_df=df[df['Sort'+str(i)+'_mean_score'].notnull()]
        if not assay_df.empty:
            for j in range(len(assay_df)):
                cat_var.append(OH_matrix[OH_counter].tolist())
            assay_df.loc[:,'y']=assay_df['Sort'+str(i)+'_mean_score']
            exploded_df.append(assay_df)
        OH_counter=OH_counter+1

    exploded_df=pd.concat(exploded_df,ignore_index=True)
    y=exploded_df.loc[:,'y'].values.tolist()

    return exploded_df, cat_var, y

def mix_with_cat_var(x_a,cat_var):
    if len(cat_var[0])>1:
        x=[]
        for i in range(len(x_a)):
            x.append(x_a[i]+cat_var[i])
        return x
    else:
        return x_a #if there is only one catagory of 'y', dont include catagorical variable in model input

def get_ordinal(df):
    'sets ordinal encoded sequence to x_a in df for use in embedding models'
    x_a=df.loc[:,'Ordinal'].values.tolist()
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    return x_a


def get_onehot(df):
    'sets one_hot encoded sequence to x_a in df'
    x_a=df.loc[:,'One_Hot'].values.tolist()
    for i in range(len(x_a)):
        x_a[i]=x_a[i].tolist()
    return x_a

def get_assays(assays,df):
    'sets assay values (of assays) to x_a in df'
    column_names=[]
    for i in assays:
        column_names.append('Sort'+str(i)+'_mean_score')
    x_a=df.loc[:,column_names].values.tolist()
    return x_a

def get_seq_and_assays(assays,df):
    'x_a is concat(onehot,assay scores)'
    assay_list=get_assays(assays,df)
    sequence_list=get_onehot(df)
    x_a=[]
    for i,j in zip(assay_list,sequence_list):
        x_a.append(i+j)
    return x_a

def get_control(df):
    'x_a should be null for a zero_rule model that guesses based upon average'
    x_a=[[]]*len(df)
    return x_a

def get_embedding(df):
    'gets learned embedding for x_a'
    x_a_in=df.loc[:,'learned_embedding']
    x_a=[[]]*len(df)
    for i in range(len(x_a)):
        x_a[i]=x_a_in.iloc[i][0].tolist()
    return x_a