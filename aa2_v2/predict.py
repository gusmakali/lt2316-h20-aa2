import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .trainer import parse_device_string
import torch.nn.functional as F

def predict(test_X, test_y, model_class, model_state_dict, optimizer_state_dict, trained_hyperparamaters, vocab, id2ner):
    input_size = int(test_X.max())+3 # len(vocab)
    batch_size = 1
    n_layers = trained_hyperparamaters['number_layers']
    bidirectional = trained_hyperparamaters['bidirectional']
    dropout = trained_hyperparamaters['dropout']
    device = parse_device_string(trained_hyperparamaters['device'])
    emb_dim = trained_hyperparamaters['emb_dim']
    hidden_dim = trained_hyperparamaters['hidden_dim']
    lr = trained_hyperparamaters['learning_rate']

    model = model_class(input_size, emb_dim, hidden_dim, 5, n_layers, bidirectional, dropout)
    model.load_state_dict(model_state_dict)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    opt.load_state_dict(optimizer_state_dict)

    model.to(device)
    h = model.init_hidden(batch_size)

    model.eval()
    rand_idx = np.random.randint(1, len(test_X), size=1)

    with torch.no_grad():
        X = test_X[rand_idx]
        y = test_y[rand_idx]

        h = tuple([each.data for each in h])
        scores, h = model(X, h)
    
    
    preds = torch.sigmoid(scores)
    preds = preds.view(preds.size(0), preds.size(2), preds.size(1))[0] 

    sent = decode_sent(X, vocab)
    reverse_sent = sent.copy()
    reverse_sent.reverse()
    padding = 0
    for w in reverse_sent:
        if w is None:
            padding +=1
        else: 
            break
    
    sent = sent[:-padding]
    sent = [w if w is not None else '<UNK>' for w in sent]

    print('SENTENCE:', " ".join(sent))
    print(")___________________")
    print()

    target = [ner_id for ner_id in y[0]]
    target = target[:-padding]
    preds = F.softmax(preds, dim=0)
    _, max_idx = torch.max(preds, dim=0)

    for i, t in enumerate(target):
        
        if target[i] == max_idx[i].long():
            predicted_ner = max_idx[i]
            token_idx = i
   
            print(f'CORRECT: predicted NER "{id2ner[predicted_ner]}" for "{sent[token_idx]}"')

        elif target[i] != max_idx[i].long():

            predicted_ner = max_idx[i]
            token_idx = i
            print(f'NOT CORRECT: predicted NER "{id2ner[predicted_ner]}" for "{sent[token_idx]}", but it was "{id2ner[t]}" ')



def decode_sent(test_X, vocab):
    enc_sent =[]
    # extracting only vocab_id column
    for w in test_X[0]:
        enc_sent.append(int(w[0]))
    decod_sent = []
    for i in enc_sent:
        decod_sent.append(vocab[i])
    return decod_sent


    
    
  
        

    
 
        
    
 