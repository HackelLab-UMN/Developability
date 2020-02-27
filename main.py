import model_training_functions as train
import pandas as pd
import hyperopt_model
import test_final_set

def main():

    #List of possible sequence-to-yield models
    model_names=['emb_rnn_atn','emb_rnn','emb_cnn_atn','emb_cnn','emb_fnn_flat','emb_fnn_max','seq_fnn_flat','emb_ridge','seqridge','seqforest','seqsvm','seqfnn_small','seqemb_fnn_small','controlridge']

    #Sample of assay-to-yield models
    #j=(1,8,9,10) #set which assays should be used to predict yield
    #assays=[] #must pass in to help format input of model
    #for i in list(j):
    #    assays.append('Sort'+str(i)+'_mean_score') #name of column in dataframe
    #model_name='assay_ridge_'+str(list(j))

    #Sample of seq-to-assay (2)
    #model_names=['2seqridge']

    import sys
    combo_no=int(sys.argv[1]) #choose which model to work on based upon MSI job array ID#

    for i in range(25): #optimize hyper-parameters, in a loop so it saves trials between runs incase MSI time runs out
        hyperopt_model.hyp_train(model_names[combo_no],assays=None)
    

    cv=test_final_set.cross_validate_best_model(model_names[combo_no],assays=None,resolve=True) #once best hyper-parameter is found, redo once to make figure
    print(model_names[combo_no])
    print(cv)
    test=test_final_set.test_final(model_names[combo_no],assays=None) #using best hyper-parameter, predict testset
    print(test)

if __name__ == '__main__':
    main()
        