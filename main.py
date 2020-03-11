import sys
import submodels_module as modelbank


def main():
    '''
    current model options are [ridge,forest,svm,fnn,emb_fnn_flat,emb_fnn_maxpool]
    '''

    toggle_no=int(sys.argv[1])
    ### seq_to_yield model using measured yields
    a_models=['ridge','forest','svm']
    a=modelbank.seq_to_yield_model(a_models[toggle_no],1)
    a.cross_validate_model()
    a.test_model()
    a.plot()

    # ### assay_to_yield_model
    # b=modelbank.assay_to_yield_model([1,8,9,10],'forest',1)
    # b.cross_validate_model()
    # b.test_model()
    # b.plot()
    # b.save_predictions() 

    ### use predictions of yield from model a to make a seq-to-(predicted)yield model
    # assay_to_yield_model_no=0 #for each saved model from a
    # c=modelbank.seq_to_pred_yield_model([[1,8,9,10],'forest',1,assay_to_yield_model_no],['emb_fnn_maxpool',0.01])
    # c.cross_validate_model()
    # c.test_model()
    # c.plot()

    ### create a sequence to assay model
    # d=modelbank.seq_to_assay_model([1,8,9,10],'emb_fnn_maxpool',0.01)
    # d.cross_validate_model()
    # d.test_model()
    # d.plot()
    # d.save_predictions() 
    # d.save_sequence_embeddings()

    ###use assay predictions of test set and assay_to_yield model 
    # seq_to_assay_model_no=0
    # b.apply_predicted_assay_scores(['emb_fnn_maxpool',0.01,seq_to_assay_model_no])

    ###use sequence embeddings to predict yield
    # seq_to_assay_model_no=0
    # e=modelbank.sequence_embeding_to_yield_model([[1,8,9,10],'emb_fnn_maxpool',0.01,seq_to_assay_model_no],'forest',1)
    # e.cross_validate_model()
    # e.test_model()
    # e.plot()


if __name__ == '__main__':
    main()