import sys
import submodels_module as modelbank
from itertools import combinations


def main():

    toggle_no=int(sys.argv[1])


    c_models=['ridge','fnn','emb_fnn_flat','emb_fnn_maxpool','emb_fnn_maxpool_linear','emb_rnn','small_emb_rnn','small_emb_atn_rnn','small_emb_rnn_linear',
        'emb_cnn','small_emb_cnn','small_emb_atn_cnn','small_emb_cnn_linear']
    c=modelbank.seq_to_assay_model([1,8,10],c_models[toggle_no],1)
    c.cross_validate_model()
    c.test_model()
    c.save_predictions()
    c.save_sequence_embeddings()





if __name__ == '__main__':
    main()