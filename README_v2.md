## LT2316 H20 Assignment A2 : Ner Classification

Name: *Liliia Makashova* 

## Notes on Part 1.

***  GENERAL NOTE ***

THIS IS THE VERSION_2 OF AA2. BOTH aa2 and aa2_v2 work fine. This readme concerns the folder aa2_v2. The caclulation of accuracy score here was changed. The `model.py` was not changed in `aa_v2` folder.

`run_v2.ipynb` (with imports from the `aa_v2` folder) should be outside `aa1` and `aa2_v2` folders. The models will be saved to the `aa2_v2/trained_models` folder. `aa1` folder was updated (a few new lines of code) according to the feedback to A1 task (the vocab is now containing only the words from the training data). Please, clone the whole repo. 

*Model of choice LSTM*

The current version of the task is running with an LSTM model. The reason for sticking with this one was performance, as the fisrt choice was GRU, but it showed lower performance in terms of accuracy. 

The model is in essence basic, with an embedding layer, droupout, linear activation and intitializied hidden & cell states to zeros.
The embedding layer is not neccessary and another implementation of the model (without `embed` is also possible), but now in the forward pass some reshaping takes place to ajust 3D data tensor into 2D tensor (as `nn.Embedding` expects a 2D tensor).

See part 3 on how hypermaramaters are used to tune the model's performance (e.g. one/multi layered, one/bidirectional etc.)

## Notes on Part 2.

*Design decisions:*

- the metric of choice - accuracy.
- training, validation and testing loops are in most implemented as usual (see caveats below).
- training and validation (`train` method in `Trainer` class) will produce print statements per epoch, printing train loss, val loss and val accuracy. Also will produce a print statement after finishing training and validating each model. 
- `test` method in `Trainer` class will print overall test accuracy and test loss. 
-  a helper method `calc_accuracy` handles the measuring for accuracy. It is used in the validation loop (the method is used to print accuacy per batch) and in the testing loop (to print accuracy for the whole test set).
- `scores` dictionary also saves aggregated validation loss (per val set).
- `save_model` saves aggregated training loss (per whole train set) as well.
- `model_name` is assigned after every training is finished by the pattern "RRN + n_layers + batch_size"

***ACCURACY CAVEATS***

- it was very tricky to establish a good way to get accuracy metrics.
- several approaches were tested. The second version of the task is simply using softmax approach.

In `calc_accuracy` at the level of one sentence, softmax is applied with regard to columns (ner_ids) - e.g. dim=0 (see the comments in the code).
Then having probabilities per token over 5 ner_ids I am taking the score that is the biggest as well as its index. The index array (the array of length 165 with values in range from 0 to 4) from torch.max is then compared to the target y_label. If predictions are the same as in target, they are considered correct ones. 

After collecting `n_correct` tokens over batches, we divide this number by the number of all tokens that were seen to get the accuracy. 

## Notes on Part 3.

*Design and tuning*

There were a few modifications to the commented template that we were given.

- optimizer and epochs are not in the `hp` dictionary: I decided that it's ok and fair to train all the models with the same optimizer and number of epochs that are initialized in the `train` funtion, as it would not make sense to compare one model trained over 3 epochs with another one trained over 17 epochs (same logic for optimizer and loss). Also for me it doesn't make sence to plot a graph over opt as `hp`.

- however, `device` is a member of `hp` as it was the only way to access the device settings from the jupyter notebook as we were not allowed to change `train` and `test` functions' parameters (still doesn't make sense to plot of over device as `hp`).

***PERFORMANCE***

I tested 8 diffferent sets of `hp` and the logic of tuning the best model was:
- after looking at the result of models trained from hp_set1, the best one (by accuracy and loss) was the one with `hp` # 3:
{
                          "learning_rate": 0.01,
                          "number_layers": 3,
                          'batch_size' : 30, 
                          'dropout': 0.25,
                          'bidirectional': True,
                          'device' :  f"{device.type}:{device.index}",
                          'hidden_dim' : 64,
                          'emb_dim': 300
                        }

In hp_set2 I decided to tune this one (hp #3) in three different combitations. 
I decided to keep relatively big emb_dim size and tried to make it even bigger (as we have big embedding input). 
I also made dropout bigger in one `hp` and smaller in two others to see how it affects the result, as this one (RNN_nl3_30) seemed not to converge and might be overfitting over more epochs. 

I kept hidden_size untouched for set2 as 64 seems to fit the data size well (after the embedding layer) and experimented a bit with learning rate by making it even smaller, so set2 models can spend more time to learn the patterns. 
Small batch_size (10 or 30) was kept. 

None of the models were trained long enough to converge due to time limitation, but in this case 11 epoch show some general pattern for all of them (see prints in the notebook).

I tested and checked the accuracy function many times. In version_2 the accuracy scores are even higher. 

## Notes on Part 4.

*Testing discussion*


The best model 'RNN_nl2_10.pt' has `hp`: 
{
                          "learning_rate": 0.001,
                          "number_layers": 2,
                          'batch_size' : 10, 
                          'dropout': 0.1,
                          'bidirectional': True,
                          'device' :  f"{device.type}:{device.index}",
                          'hidden_dim' : 64,
                          'emb_dim': 700
                        }

In the best model - smaller lr, bigger emb_dim and smaller dropout performed much better - stable and higher accuracy, however val_loss neither shrinks nor grows (needs more epochs to converge).

The worse model reaches in average around 70% accuracy score over test set (see the plot from set1). 

In general, the small dropout seems to perform better, and bigger one seems to be leading to overfitting, that's what the print statements for val_loss signal about in several models. 

The best model stays at around 98% in accuracy over the val set, and gives 98,27% on the test set. The loss seems to shrink during the training loop, but model is not converging yet at 11 epochs.


## Notes on Part Bonus.

Please see my `predict` function. It's imported in run_v2.ipynb from an updated `predict.py`.
It will randomly choose a sentence from the test test. It will make a prediction on it and give an extensive print statement per token whether the prediction was correct. If it was not correct it will tell which one is correct.

***Further discussion and BIO***  
As you can see from the print of `predict` my model is almost 100% correct about recognizing '0' - non-NERs. 
I know that 0 label is 70% of the whole data, however, I cannot still understand how I am getting such a hight accuracy (if you manage to find a bug PLEASE point me to it). 

I decided to manually label with BIO, just because it seems to be the simplest to encode (it wouldn't take much effort at this stage to re-encode my labels to this format) and therefore, for the model to learn. The BIO method does not cover the edge cases as below. However, such are only few in our data, so maybe there is no point to make a complicated encoding that might confuse other predictions. The sentece (below) got correct labels for 0, but not for NERS.

id2ner = [None, 'group', 'drug_n', 'drug', 'brand']

```
The  treatment of coinfected patients requires antituberculosis and antiretroviral drugs to be administered concomittantly  ; 
0       0       0       0       0       0           B-1          0      B-1         i-1  0   0         0        0           0 

antituberculosis drugs - NER group
antiretroviral drugs - NER group
```

As mentioned before, this sentence from the test set is an edge case and such encoding does not show that 'anituberculosis' is not a single NER but also has 'drug' as part of this discontinuous entity. 


*** As a note *** 

I relized that I was supposed to make tokenization more severe, e.g. removing all non-words (.,.?...) so I would shrink the amount of zeros in the data and possibly make a better model. 