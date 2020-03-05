import submodels_module as modelbank


def main():
    '''pick your favorite one to try, initialize with model architecture and % of dataset, and assays if needed
    Eventually I'll need to write a script to convert array ID into diffferent models.

    current model options are [ridge,forest,svm]
    '''

    # assay_to_yield_model(list of assays, model_arch, % data)
    a=modelbank.assay_to_yield_model([1,2,3],'ridge',0.5)
    a.cross_validate_model()
    a.test_model()

    #seq_to_assay_model(list of assays, model_arch, % data)
    b=modelbank.seq_to_assay_model([1,2,3],'ridge',0.01)
    b.cross_validate_model()
    b.test_model()

    # seq_to_yield_model(model_arch, % data)
    c=modelbank.seq_to_yield_model('ridge',0.5)
    c.cross_validate_model()
    c.test_model()

    # control_to_yield_model(model arch, %)
    d=modelbank.control_to_yield_model('ridge',1)
    d.cross_validate_model()
    d.test_model()

    # control_to_assay_model(assays,model arch, %)
    e=modelbank.control_to_assay_model([1,2,3],'ridge',0.01)
    e.cross_validate_model()
    e.test_model()


if __name__ == '__main__':
    main()