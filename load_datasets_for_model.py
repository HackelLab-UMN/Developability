
import pandas as pd
pd.options.mode.chained_assignment = None
import numpy as np

def OH_encode_yields(dataset): #need to split sequences with both IQ and SH yields
    IQ_data=dataset[dataset['IQ_Average_bc'].notnull()]
    if not IQ_data.empty:
        IQ_data.loc[:,'IQ_binary']=1
        IQ_data.loc[:,'SH_binary']=0
        IQ_data.loc[:,'Yield']=IQ_data['IQ_Average_bc']
        IQ_data.loc[:,'Yield_std']=IQ_data['IQ_Average_bc_std']

    SH_data=dataset[dataset['SH_Average_bc'].notnull()]
    if not SH_data.empty:
        SH_data.loc[:,'IQ_binary']=0
        SH_data.loc[:,'SH_binary']=1
        SH_data.loc[:,'Yield']=SH_data['SH_Average_bc']
        SH_data.loc[:,'Yield_std']=SH_data['SH_Average_bc_std']

    return IQ_data.append(SH_data,ignore_index=True,sort=False)

def load_x_to_yield_data(dataset):
    #Used for predicting test set of just predictive assay scores
    # assay_scores=[]
    # for i in [1,5,8,9,10]:
    #     assay_scores.append('Sort'+str(i)+'_mean_score')
    # dataset=dataset[~dataset[assay_scores].isna().any(axis=1)] 
    # print(len(dataset))

    dataset=OH_encode_yields(dataset)
    return dataset #return whole dataset so 'learned' embeddings are also transfered

def OH_encode_assays(dataset):
    dataset_out=[]
    used_assays=[1,8,9,10] #these are the assays to be used to train embedding layers
    assay_OH=np.eye(len(used_assays)) #one_hot vectors for each of the used assays
    for i,j in zip(used_assays,assay_OH):
        assay_name='Sort'+str(i)+'_mean_score'
        assay_std_name='Sort'+str(i)+'_std_score'
        dataset_assay=dataset[dataset[assay_name].notnull()]
        if not dataset_assay.empty: #if assay score exists, add it to training dataset
            dataset_assay.loc[:,'Assay_ID']=[[j]]*len(dataset_assay)
            dataset_assay.loc[:,'Score']=dataset_assay[assay_name]
            dataset_assay.loc[:,'Score_std']=dataset_assay[assay_std_name]
        dataset_out.append(dataset_assay)
    dataset_out=pd.concat(dataset_out,ignore_index=True)
    return dataset_out

def load_seq_to_assay_data(dataset):
    dataset=OH_encode_assays(dataset)

    seq_OH=dataset['One_Hot']
    seq_ord=dataset['Ordinal']
    assay_OH=dataset['Assay_ID']
    score_=dataset['Score']
    score_std=dataset['Score_std']


    data=pd.concat([seq_OH,seq_ord,assay_OH,score_,score_std],axis=1)
    return data




