import submodels_module as modelbank


def main():
    '''pick your favorite one to try, initialize with model architecture and % of dataset, and assays if needed
    Eventually I'll need to write a script to convert array ID into diffferent models.

    current model options are [ridge,forest,svm]
    '''

    # assay_to_yield_model(list of assays, model_arch, % data)
    # a=modelbank.assay_to_yield_model([1,8,9,10],'forest',1)
    # a.cross_validate_model()
    # a.test_model()
    # a.plot()
    # a.save_predictions()

    a2=modelbank.seq_to_pred_yield_model([[1,8,9,10],'forest',1,0],['forest',0.01])
    a2.cross_validate_model()
    a2.test_model()
    a2.plot()

    #seq_to_assay_model(list of assays, model_arch, % data)
    # b=modelbank.seq_to_assay_model([1,2,3],'ridge',0.01)
    # b.cross_validate_model()
    # b.test_model()
    # b.plot()

    # seq_to_yield_model(model_arch, % data)
    # c=modelbank.seq_to_yield_model('forest',1)
    # c.cross_validate_model()
    # c.test_model()
    # c.plot()

    # # control_to_yield_model(model arch, %)
    # d=modelbank.control_to_yield_model('ridge',1)
    # d.cross_validate_model()
    # d.test_model()
    # d.plot()

    # # control_to_assay_model(assays,model arch, %)
    # e=modelbank.control_to_assay_model([1,2,3],'ridge',0.01)
    # e.cross_validate_model()
    # e.test_model()


if __name__ == '__main__':
    main()