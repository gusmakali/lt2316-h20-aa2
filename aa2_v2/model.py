import torch.nn as nn
from torch import optim
import torch

class RNN_model(nn.Module):
    def __init__(self, input_size, emb_dim, h_dim, output_dim, num_layers, bidirectional, dropout):
        super(RNN_model, self).__init__()
        self.num_layers = num_layers
        self.h_dim = h_dim
        self.bidirectional = bidirectional

        self.embedd = nn.Embedding(input_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_size=h_dim,
                            num_layers=num_layers,
                            batch_first=True, 
                            bidirectional=bidirectional,
                            dropout=dropout)
        
        self.predict = nn.Linear(h_dim*2 if bidirectional else h_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, sentence, hidden_st):

        # since we have a 3d tensor as input and embeddings layer expect 2d tensor
        # some reshaping is needed

        emb_sent = sentence.view(-1, sentence.size(2))
        emb = self.embedd(emb_sent)
        emb = emb.view(*sentence.size(), -1)
        emb = emb.sum(2)

        out, hidden_st = self.rnn(emb, hidden_st)
        out = self.dropout(out)
        scores = self.predict(out)
        
        return scores, hidden_st

    def init_hidden(self, batch_size):

        w = next(self.parameters()).data

        if self.bidirectional:
            hidden_st = (w.new(self.num_layers*2, batch_size, self.h_dim).zero_(),
                        w.new(self.num_layers*2, batch_size, self.h_dim).zero_())
        else:
            hidden_st = (w.new(self.num_layers, batch_size, self.h_dim).zero_(),
                        w.new(self.num_layers, batch_size, self.h_dim).zero_())

        return hidden_st
  
        
        
        

