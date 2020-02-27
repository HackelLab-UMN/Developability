import load_datasets_for_model as loadmodel
import plotting_functions
import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_squared_error as mse
from functools import partial
from parallel_training import parallel_part


#this function evaluates model in both CV and final mode, there are some common parts, but I need to figure out how to combine them 
def evaluate_model(dataset,space,model_name,assays,cv,save_fig=False):
    if cv: #cross validation evaluations
        parallel_part_model=partial(parallel_part,space,model_name,assays,False) #save_model is false during CV
        if '2' in model_name:
            n_splits=3
        else:
            n_splits=10
        
        #split datasets for cross-validatoin
        kf=RepeatedKFold(n_splits=n_splits,n_repeats=cv)
        train,test=[],[]
        for train_index, test_index in kf.split(np.zeros(len(dataset))):
            train.append(dataset.iloc[train_index])
            test.append(dataset.iloc[test_index])
        cv_sets=zip(train,test)

        #train models via training set, collect loss of test set
        if 'nn' in model_name: #tensorflow GPU doesn't work with multiprocessing
            data=[]
            for i in cv_sets:
                data.append(parallel_part_model(i))
        else: #svm, ridge, forest can be done in parallel for increased speed
            from multiprocessing import Pool
            pool=Pool(processes=24)
            data=pool.map(parallel_part_model,cv_sets)
            pool.close()
            pool.join()

        # combine data from each model 
        data=np.array(data)
        train_var,train_err,test_var,test_err=[],[],[],[]
        for i in range(0,len(data),n_splits): #calculate average across splits of a single CV
            train_var.append(np.mean(data[i:i+n_splits,0]))
            train_err.append(np.mean(data[i:i+n_splits,1]))
            test_var.append(np.mean(data[i:i+n_splits,2]))
            test_err.append(np.mean(data[i:i+n_splits,3]))

        CV_data=np.array([train_var,train_err,test_var,test_err]).mean(axis=1) #calculate average across CV's
        CV_std=np.array(test_err).std()
        CV_data=np.append(CV_data,CV_std)

        if cv==1:  # if CV is one, then we can plot the predicitions for the dataset
            true_test=np.concatenate(data[i:i+n_splits,4])
            predicted_test=np.concatenate(data[i:i+n_splits,5])
            test_var_plot=CV_data[2]
            test_err_plot=CV_data[3]
            if save_fig:
                fig_name='./'+model_name+'_CV.png'
                if '2'in model_name:
                    fig=plotting_functions.plot_assay_predictions(true_test,predicted_test,model_name,test_var_plot,test_err_plot,fig_name)
                else:
                    fig=plotting_functions.plot_yield_predictions(true_test,predicted_test,model_name,test_var_plot,test_err_plot,fig_name)
        return CV_data #[training exp variance, training mse, test exp variance, test mse, std across cv's of test mse]            

    else:
        #train/save model using best space and all of data. Test on independent test-set (different than CV-test set)
        train=dataset #use the entire dataset to train
        if '2' in model_name:
            test=pd.read_pickle('./assay_to_dot_training_data.pkl')
            save_model=3
        else:
            test=pd.read_pickle('./seq_to_dot_test_data.pkl')
            save_model=3

        test_repeats=[]
        for model_no in range(1,save_model+1): #if model_no==0, then it won't save model
            parallel_part_model=partial(parallel_part,space,model_name,assays,model_no) #during testing we want to save final model 
            _,_,test_var,test_err,true_test,predicted_test=parallel_part_model([train,test])
            test_repeats.append([test_var,test_err])
            if model_no==1: #save a plot of predictions for the test set of the first model 
                fig_name='./'+model_name+'_test.png'
                if '2' in model_name:
                    fig=plotting_functions.plot_assay_predictions(true_test,predicted_test,model_name,test_var,test_err,fig_name)
                else:
                    fig=plotting_functions.plot_yield_predictions(true_test,predicted_test,model_name,test_var,test_err,fig_name)

        test_repeats=np.array(test_repeats)
        test_err=test_repeats[:,1].mean()
        test_std=test_repeats[:,1].std()
        test_var=test_repeats[:,0].mean()
       
        return test_err,test_std,test_var
