
#basics
import pandas as pd
import nltk
import numpy as np
import torch
import pickle
import os.path

# Feel free to add any new code to this script

def add_features(data_df, id2word):
    
    decode_word = lambda x: str(id2word[x]) if x < len(id2word) else '' # real word  --> vocab [7000]     
    data_df['token'] = data_df.loc[:,'token_id'].apply(decode_word)

    pos_tag = lambda x: nltk.pos_tag([x])[0][1] if len(x) > 0 else ''
    data_df['pos'] = data_df.loc[:, 'token'].apply(pos_tag)

    get_prefix = lambda x: x[:4]
    data_df['prefix'] = data_df.loc[:, 'token'].apply(get_prefix)

    get_suffix = lambda x: x[-3:]
    data_df['suffix'] = data_df.loc[:, 'token'].apply(get_suffix)

    get_len = lambda x: len(x)
    data_df['token_length'] = data_df.loc[:, 'token'].apply(get_len)

    return data_df 

def encode_features(data_df, vocab):
    unique_pos = data_df['pos'].unique().tolist()
    # unique_pos.remove('')
    id2pos = [''] + unique_pos
    unique_prefix = data_df['prefix'].unique().tolist()
    # unique_prefix.remove('')
    id2prefix = [''] + unique_prefix
    unique_suffix = data_df['suffix'].unique().tolist()
    # unique_suffix.remove('')
    id2suffix = [''] + unique_suffix
    
    encode_token = lambda x: vocab.index(x) if x in vocab else 0
    data_df['vocab_id'] = data_df.loc[:, 'token'].apply(encode_token)

    encode_pos = lambda x: id2pos.index(x)
    data_df['pos'] = data_df.loc[:, 'pos'].apply(encode_pos)

    encode_suf = lambda x: id2suffix.index(x)
    data_df['suffix'] = data_df.loc[:, 'suffix'].apply(encode_suf)

    encode_pref = lambda x: id2prefix.index(x)
    data_df['prefix'] = data_df.loc[:, 'prefix'].apply(encode_pref)
    
    return data_df, id2pos, id2prefix, id2suffix

def extract_features(data:pd.DataFrame, max_sample_length:int, id2word, vocab, device):
    # this function should extract features for all samples and 
    # return a features for each split. The dimensions for each split
    # should be (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH, FEATURE_DIM)
    # NOTE! Tensors returned should be on GPU
    #
    # NOTE! Feel free to add any additional arguments to this function. If so
    # document these well and make sure you dont forget to add them in run.ipynb
    # print(id2word)

    df_features = add_features(data, id2word)
    df_features_enc, _, _, _ = encode_features(df_features, vocab)

    df_features_enc_train = df_features_enc[df_features_enc['split'] == 'TRAIN']
    df_features_enc_val = df_features_enc[df_features_enc['split'] == 'VAL']

    df_features_enc_val.loc[df_features_enc_val['vocab_id'] == 0, ['pos', 'prefix', 'suffix', 'token_length']] = 0

    df_features_enc_test = df_features_enc[df_features_enc['split'] == "TEST"]

    df_features_enc_test.loc[df_features_enc_test['vocab_id'] == 0, ['pos', 'prefix', 'suffix', 'token_length']] = 0

    # print(df_features_enc_val[df_features_enc_val['vocab_id'] != 0])
    # print(df_features_enc_test[df_features_enc_test['vocab_id'] != 0])

    def get_array(df_enc):

        grouped_by_sent = df_enc.groupby(['sentence_id'])
        X_array = []

        for sent_id, group in grouped_by_sent:
            
            features = [[row['vocab_id'], row['pos'], row['prefix'], row['suffix'], row['token_length']] for i, row in group.iterrows()]
            if len(features) > max_sample_length:
                features = features[:max_sample_length]
            else:
                features += [[0,0,0,0,0]] * (max_sample_length-len(features))
            X_array.append(features)

        X_array = np.asarray(X_array)
    
        return X_array
    
    X_train = get_array(df_features_enc_train)
    X_val = get_array(df_features_enc_val)
    X_test = get_array(df_features_enc_test)

    X_train_tensor= torch.from_numpy(X_train)
    X_val_tensor = torch.from_numpy(X_val)
    X_test_tensor = torch.from_numpy(X_test)

    REPICKLE = False
    PICKLE_DATA_FILE = 'tensor.pickle'

    if REPICKLE == True or not os.path.isfile(PICKLE_DATA_FILE):
        df_files = open(PICKLE_DATA_FILE, 'wb')
        pickle.dump((X_train_tensor, X_val_tensor, X_test_tensor), df_files)
        df_files.close()
    else:
        
        df_files = open(PICKLE_DATA_FILE, 'rb')
        X_train_tensor, X_val_tensor, X_test_tensor = pickle.load(df_files)


    return X_train_tensor.to(device), X_val_tensor.to(device), X_test_tensor.to(device) 

