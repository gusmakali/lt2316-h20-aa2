
import os
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F


class Data(Dataset):
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
    def __len__(self):
        return len(self.X_data)

class Trainer:

    def __init__(self, dump_folder="/aa2/trained_models"):
        self.dump_folder = dump_folder
        os.makedirs(dump_folder, exist_ok=True)


    def save_model(self, epoch, model, optimizer, loss, scores, hyperparamaters, model_name):
        # epoch = 1
        # model =  model.state_dict()
        # optimizer = torch.optim.Adam(model_CNN.parameters(self), lr=0.001).state_dict()
        # loss = nn.BCEWithLogitsLoss()
        # scores = scores
        # hyperparamaters = hyperparamaters
        # model_name = 'model_GRU'
        #  
        #
        # More info about saving and loading here:
        # https://pytorch.org/tutorials/beginner/saving_loading_models.html#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training

        save_dict = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'hyperparamaters': hyperparamaters,
                        'loss': loss,
                        'scores': scores,
                        'model_name': model_name
                        }

        torch.save(save_dict, os.path.join(self.dump_folder, model_name + ".pt"))


    def load_model(self, model_path):
        # Finish this function so that it loads a model and return the appropriate variables

        checkpoint = torch.load(model_path)
        epoch = checkpoint['epoch']
        model_state_dict = checkpoint['model_state_dict']
        optimizer_state_dict = checkpoint['optimizer_state_dict']
        hyperparamaters = checkpoint['hyperparamaters']
        loss = checkpoint['loss']
        scores = checkpoint['scores']
        model_name = checkpoint['model_name']

        return epoch, model_state_dict, optimizer_state_dict, hyperparamaters, loss, scores, model_name
    


    def train(self, train_X, train_y, val_X, val_y, model_class, hyperparamaters):
        # Finish this function so that it set up model then trains and saves it.
        # GLOBAL PARAMETERS (NOT CHANGABLE)
        # clip = 5  # for gradient during training
        input_size = int(train_X.max()) + 1  # a.k.a. length of vocab 
        criterion = nn.CrossEntropyLoss()
        epochs = 11

        # created batched data
        train_load = self.get_batch_data(train_X, train_y, hyperparamaters)
        val_load = self.get_batch_data(val_X, val_y, hyperparamaters)
    
        # extract hyperparamaters 
        lr = hyperparamaters['learning_rate']
        batch_size = hyperparamaters['batch_size']
        n_layers = hyperparamaters['number_layers']
        dropout = hyperparamaters['dropout']
        device = parse_device_string(hyperparamaters['device'])
        emb_dim = hyperparamaters['emb_dim']
        hidden_dim = hyperparamaters['hidden_dim']
        bidirectional = hyperparamaters['bidirectional']
        
        model = model_class(input_size, emb_dim, hidden_dim, 5, n_layers, bidirectional, dropout)

        # all models use same optimizer
        opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        
        model.to(device)
        model.train()

        print(model)
        print("Training... \n")


        agg_loss = []
        agg_acc = []

        for e in range(epochs):

            # initializing hidden state
            h = model.init_hidden(batch_size)

            loss_sum = 0
            n_batches = 0

            for batch in train_load:
                
                # for hidden and cell states
                h = tuple([each.data for each in h])
                X_batch, y_batch = batch

                preds, h = model(X_batch, h)
                
                opt.zero_grad()

                # flip prediction tensor for loss function 
                preds = preds.view(preds.size(0), preds.size(2), preds.size(1))

                loss_score = criterion(preds, y_batch)
                loss_score.backward()

                nn.utils.clip_grad_norm_(model.parameters(), 5)   
                opt.step()
                
                loss_sum += loss_score.item()
                n_batches += 1

            train_loss = loss_sum / n_batches                

            agg_loss.append(train_loss)

            valid_loss = 0
            n_correct = 0
            # since calc_accuracy measures accuract per token (not sentence)
            # we need to get number of all tokens in a set
            corr_val = len(val_load.dataset) * 165
            loss_sum = 0
            n_batches = 0

            valid_h = model.init_hidden(batch_size)
            model.eval()

            for batch in val_load:
                valid_h = tuple([each.data for each in valid_h])
                X_batch, y_batch = batch

                preds, valid_h = model(X_batch, valid_h)
                preds = preds.view(preds.size(0), preds.size(2), preds.size(1))

                n_corr_batch, loss_batch = self.calc_accuracy(preds, criterion, y_batch.float())
                loss_sum += loss_batch
                n_correct += n_corr_batch
                n_batches += 1

                
            valid_acc = n_correct / corr_val
            agg_acc.append(valid_acc)
            valid_loss += loss_sum / n_batches
           
            print(f"Epoch {e+1:03}: training loss = {train_loss:.4f} | validation loss = {valid_loss:.4f} | validation accuracy: {valid_acc*100:.2f}%")
           

            # reset to training again
            model.train()
            
        scores = {
                    'accuracy': (sum(agg_acc)/len(agg_acc)) * 100
                    }

        agg_train_loss = sum(agg_loss) / len(agg_loss)
       
        self.save_model(epochs, model, opt, agg_train_loss, scores, hyperparamaters, f'RNN_nl{n_layers}_{batch_size}')
        
        print(f'Done training RNN_nl{n_layers}_{batch_size}! \n')
        print('___________________\n')



    def test(self, test_X, test_y, model_class, best_model_path):
        # Finish this function so that it loads a model, test is and print results.
        trained_epochs, model_state_dict, optimizer_state_dict, trained_hyperparamaters, trained_loss, trained_scores, model_name = self.load_model(best_model_path)

        test_load = self.get_batch_data(test_X, test_y, trained_hyperparamaters, True)


        # GLOBAL PARAMATERS
        criterion = nn.CrossEntropyLoss()
        # input size is common for all models, so it should be len(vocab)
        # since we don't have train_X or vacab in this function
        # I established input_size in an experimental way 
        # meaning that test set doesn't have a maximum vocab_id inside
        input_size = int(test_X.max()) + 3


        # get hyperparam
        batch_size = trained_hyperparamaters['batch_size']
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


        n_correct = 0
        corr_test = len(test_load.dataset) * 165
        n_batches = 0
        loss_sum = 0 
        test_loss = 0
    

        for batch in test_load:
            h = tuple([each.data for each in h])

            with torch.no_grad():
                X_batch, y_batch = batch

                preds, h = model(X_batch, h)

                preds = preds.view(preds.size(0), preds.size(2), preds.size(1))

            n_corr_batch, loss_batch = self.calc_accuracy(preds, criterion, y_batch.float())

            loss_sum += loss_batch
            n_correct += n_corr_batch
            n_batches += 1

        test_acc = (n_correct / corr_test) * 100 
        test_loss += loss_sum / n_batches

        print(f'Test loss: {test_loss:.3f} | Test accuracy: {test_acc:.2f}%')
        return test_acc, test_loss
  
    def get_batch_data(self, X, y, hyperparamaters, is_test=False):
        batch_size = hyperparamaters['batch_size']

        if not is_test:

            data = Data(X, y)

            batched_data = DataLoader(dataset=data, batch_size=batch_size, drop_last=True, shuffle=True)
        else:

            data = Data(X, y) 

            batched_data = DataLoader(dataset=data, batch_size=batch_size, drop_last=True)
       
        return batched_data


    def calc_accuracy(self, scores, criterion, target):

        n_correct = 0
        for pred, targ in zip(scores, target):
            # dim=0 means get softmax() probability and then max_score across 5 columns (5 ner ids columns) for each token
            # then I check whether the index of max score from columns
            # equals the target array (which is values from 0 to 4),
            # which basically represents the index of ner_ids columns
            pred = F.softmax(pred, dim=0)
            max_pred_per_ner, max_ner_idx = torch.max(pred,dim=0)

            for t,p in zip(targ, max_ner_idx):
                if t == p.float():
                    n_correct +=1

        return n_correct, criterion(scores, target.long()).item()

    

def parse_device_string(device_string):
    return torch.device(device_string)