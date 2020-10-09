

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from .trainer import parse_device_string


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

    for word_idx, ner_id in enumerate(target):
        all_ner_pred = sum([torch.round(p[word_idx]) for p in preds])
        pred_by_class = [float(p[word_idx]) for p in preds]

        ner_id_max_pred = pred_by_class.index(max(pred_by_class))
        max_pred_label = id2ner[ner_id_max_pred]
        predicted_ner_correctness = torch.round(preds[ner_id][word_idx]).long()

        if predicted_ner_correctness == 1 and all_ner_pred == 1:
            print(f'CORRECT: predicted NER {max_pred_label} for "{sent[word_idx]}"')
        elif predicted_ner_correctness == 0 and all_ner_pred == 0:
            print(f'NOT CORRECT: no predictions for "{sent[word_idx]}" which is "{id2ner[ner_id]}"')
        elif predicted_ner_correctness == 0:
            print(f'NOT CORRECT: predicted NER {max_pred_label} for "{sent[word_idx]}", but it was "{id2ner[ner_id]}" ')
        else:
            ners_predicted_by_error = [i for i, pr in enumerate(pred_by_class) if round(pr) == 1]
            wrong_labels = [id2ner[n] for n in ners_predicted_by_error]
            wrong_labels.remove(max_pred_label)

            print(f'AMBIGUOUS: NER "{max_pred_label}" got max score, but label(s) {wrong_labels} was(were) also assign to "{sent[word_idx]}", which is {id2ner[ner_id]}')
        print("_____________\n")
      

def decode_sent(test_X, vocab):
    enc_sent =[]
    # extracting only vocab_id column
    for w in test_X[0]:
        enc_sent.append(int(w[0]))
    decod_sent = []
    for i in enc_sent:
        decod_sent.append(vocab[i])
    return decod_sent


    
    
  
        

    
 
        
    
 