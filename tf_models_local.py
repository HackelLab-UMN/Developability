from warnings import simplefilter 
simplefilter(action='ignore', category=FutureWarning)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
  
def fnn_small(space,input_size,model_name):
    layers=int(space['layers'])
    nodes=int(space['nodes'])
    inputs=tf.keras.Input(shape=(input_size,))

    if 'emb' in model_name:
        emb_dim=int(space['emb_dim'])
        input_seq=tf.keras.layers.Lambda(lambda x: x[:,:16])(inputs)
        embed=tf.keras.layers.Embedding(21,emb_dim,input_length=16)(input_seq) #(batch size, seq len, embed size)
        flat_seq=tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')(embed) #pool across sequence len, end with (batch size, embed size)

        input_assay=tf.keras.layers.Lambda(lambda x: x[:,16:])(inputs)
        recombined=tf.keras.layers.concatenate([flat_seq,input_assay])
    else:
        recombined=inputs

    dense=tf.keras.layers.Dense(nodes,activation='relu')(recombined)
    for i in range(layers-1):
        dense=tf.keras.layers.Dense(nodes,activation='relu')(dense)
    outputs=tf.keras.layers.Dense(1,activation='linear')(dense)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model




def seq_assay_mix(combined_layer,space,model_name):
    layers=int(space['layers'])
    nodes=int(space['nodes'])
    dense_dropout=space['dense_drop']

    dense,drop=[[]]*layers,[[]]*layers

    drop[0]=tf.keras.layers.Dropout(rate=dense_dropout)(combined_layer)
    dense[0]=tf.keras.layers.Dense(nodes,activation='relu')(drop[0])
    for i in range(1,layers):
        drop[i]=tf.keras.layers.Dropout(rate=dense_dropout)(dense[i-1])
        dense[i]=tf.keras.layers.Dense(nodes,activation='relu')(drop[i])
    if '2' in model_name:
        outputs=tf.keras.layers.Dense(1,activation='sigmoid')(drop[-1]) #assay score in range 0 to 1
    else:
        outputs=tf.keras.layers.Dense(1,activation='linear')(drop[-1]) #yield prediction with linear activation 

    return outputs


def fnn(space,input_size,model_name):

    inputs=tf.keras.Input(shape=(input_size,))

    if 'emb' in model_name:
        emb_dim=int(space['emb_dim'])
        input_seq=tf.keras.layers.Lambda(lambda x: x[:,:16])(inputs)
        embed=tf.keras.layers.Embedding(21,emb_dim,input_length=16)(input_seq) #(batch size, seq len, embed size)
        if 'flat' in model_name:
            flat_seq=tf.keras.layers.Flatten()(embed) #pool across sequence len, end with (batch size, embed size)
        elif 'max' in model_name:
            flat_seq=tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')(embed) #pool across sequence len, end with (batch size, embed size)
        normalized=tf.keras.layers.LayerNormalization()(flat_seq)
        input_assay=tf.keras.layers.Lambda(lambda x: x[:,16:])(inputs)
        recombined=tf.keras.layers.concatenate([normalized,input_assay])
    else:
        recombined=inputs

    outputs=seq_assay_mix(recombined,space,model_name)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

def cnn(space,input_size,model_name):
    emb_dim=int(space['emb_dim'])
    kernel=int(space['kernel_size'])
    filters=int(space['filters'])
    input_drop=space['input_drop']
    cov_drop=space['cov_drop']


    inputs=tf.keras.Input(shape=(input_size,))
    input_seq=tf.keras.layers.Lambda(lambda x: x[:,:16])(inputs)
    input_assay=tf.keras.layers.Lambda(lambda x: x[:,16:])(inputs)

    embed=tf.keras.layers.Embedding(21,emb_dim,input_length=16)(input_seq) #(batch size, seq len, embed size)
    drop=tf.keras.layers.Dropout(rate=input_drop,noise_shape=[None,16,1])(embed) #randomly drop of of the letters in sequence
    cov=tf.keras.layers.Conv1D(filters=filters,kernel_size=kernel,activation='relu')(drop)
    cov_drop=tf.keras.layers.Dropout(rate=cov_drop)(cov)
    if 'atn' in model_name:
        atn=tf.keras.layers.Attention()([cov_drop,cov_drop])
        flat_seq=tf.keras.layers.GlobalAveragePooling1D()(atn)
    else:
        flat_seq=tf.keras.layers.GlobalMaxPool1D(data_format='channels_last')(cov_drop) #pool across sequence len, end with (batch size, embed size)

    normalized=tf.keras.layers.LayerNormalization()(flat_seq)
    recombined=tf.keras.layers.concatenate([normalized,input_assay])
    outputs=seq_assay_mix(recombined,space,model_name)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

def rnn(space,input_size,model_name):
    emb_dim=int(space['emb_dim'])
    units=int(space['units'])
    input_drop=space['input_drop']
    recurrent_drop=space['recurrent_drop']


    inputs=tf.keras.Input(shape=(input_size,))
    input_seq=tf.keras.layers.Lambda(lambda x: x[:,:16])(inputs)
    input_assay=tf.keras.layers.Lambda(lambda x: x[:,16:])(inputs)

    embed=tf.keras.layers.Embedding(21,emb_dim,input_length=16)(input_seq) #(batch size, seq len, embed size)
    if 'atn' in model_name:
        rec=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units,recurrent_dropout=recurrent_drop,dropout=input_drop,return_sequences=True))(embed)
        atn=tf.keras.layers.Attention()([rec,rec])
        flat_seq=tf.keras.layers.GlobalAveragePooling1D()(atn)
    else:
        flat_seq=tf.keras.layers.Bidirectional(tf.keras.layers.GRU(units=units,recurrent_dropout=recurrent_drop,dropout=input_drop))(embed)
    normalized=tf.keras.layers.LayerNormalization()(flat_seq)
    recombined=tf.keras.layers.concatenate([normalized,input_assay])
    outputs=seq_assay_mix(recombined,space,model_name)
    model=tf.keras.Model(inputs=inputs,outputs=outputs)
    return model

