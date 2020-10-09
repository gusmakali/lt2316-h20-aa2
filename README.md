## LT2316 H20 Assignment A2 : Ner Classification

Name: *Liliia Makashova* 

## Notes on Part 1.

***  GENERAL NOTE ***

`run.ipynb` should be outside `aa1` and `aa2` folders. The models will be saved to the `aa2/trained_models` folder. `aa1` folder was updated (a few new lines of code) according to the feedback to A1 task (the vocab is now containing only the words from the training data). Please, clone the whole repo. 

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
- several approaches were tested. See below the technique explanation.

The model's predictions tensor for both `calc_accuracy` and loss function is shaped like this:

3 dimentions representing [batch_size, ner_labels, sent_len].

*One sentence's prediction*

- example 1:

[[[1., 1., 1.,  ..., 1., 1., 0.], # ner_id 0

[0., 0., 0.,  ..., 0., 0., 0.],   # ner_id 1

[0., 0., 0.,  ..., 0., 0., 1.],   # ner_id 2

[0., 0., 0.,  ..., 0., 0., 0.],   # ner_id 3

[0., 0., 0.,  ..., 0., 0., 0.]]   # ner_id 4

By having 5 rows of length = sent_len I am getting a pseudo-binary prediction: the current example would mean that the tokens in the begging of the sentence were classified as ner_id '0' (or None) and the last token is ner_id '2'.

However, I discovered that this is not the only possible situation. My classifier sometimes would give a result as this:

- example 2:

[[[1., 1., 1.,  ..., 1., 1., 0.], # ner_id 0

[1., 0., 0.,  ..., 0., 0., 0.],   # ner_id 1

[1., 0., 0.,  ..., 0., 0., 1.],   # ner_id 2

[1., 0., 0.,  ..., 0., 0., 0.],   # ner_id 3

[1., 0., 0.,  ..., 0., 0., 0.]]   # ner_id 4

As you can see, in this case the model cannot decide what ner_id should go for the first token. 

Therefore there are two conditions to decide whether a token is classified correctly:
1) simple one = if in the ner_id row a prediction for a token was given as '1', does is fit the y_label for this token? If yes, see 2nd condintion, if no, prediction is not coreect:
2) is there any more ones in predictions (ner_id rows) for this token: e.g. does the sum of this prediction over ner_id rows == 1? If yes, prediction is correct, if no - incorrect. 

With this logic (ONLY SINGLE PREDICTION PER TOKEN), the number for correctly classified tokens from example 1 would be 6, and from example 2 - 5 (excluding the first column with only ones).

After collecting `n_correct` tokens over batches, we devide this number by the number of all tokens that were seen to get the accuracy. 

See further explanation and discussion of the case like example 2 in part 4 (marked as 'AMBIGUOUS').

## Notes on Part 3.

*Design and tuning*

There were a few modifications to the commented template that we were given.

- optimizer and epochs are not in the `hp` dictionary: I decided that it's ok and fair to train all the models with the same optimizer and number of epochs that are initialized in the `train` funtion, as it would not make sense to compare one model trained over 3 epochs with another one trained over 17 epochs (same logic for optimizer and loss). Also for me it doesn't make sence to plot a graph over opt as `hp`.

- however, `device` is a member of `hp` as it was the only way to access the device settings from the jupyter notebook as we were not allowed to change `train` and `test` functions' parameters (still doesn't make sense to plot of over device as `hp`).

***PERFORMANCE***

I tested 8 diffferent sets of `hp` and the logic of tuning the best model was:
- after looking at the result of models trained from hp_set1, the best one (by accuracy and loos) was the one with `hp` # 3:
{
                          "learning_rate": 0.01,
                          "number_layers": 3,
                          'batch_size' : 30, 
                          'dropout': 0.25,
                          'bidirectional': True,
                          'device' :  f"{device.type}:{device.index}",
                          'hidden_dim' : 64,
                          'emb_dim': 300
                        },

In hp_set2 I decided to tune this one (hp #3) in three different combitations. 
I decided to keep relatively big emb_dim size and tried to make it even bigger (as we have big embedding input). 
I also made dropout bigger in one `hp` and smaller in two others to see how it affects the result, even when the model from hp3 was not overfitting. 
I kept hidden_size untouched for set2 as 64 seems to fit the data size well (after the embedding layer) and experimented a bit with learning rate by making it even smaller, so set2 models can spend more time to learn the patterns. 
Small batch_size (10 or 30) and multi-layerness were kept. 


None of the models were trained long enough to converge due to time limitation, but in this case 11 epoch show some general pattern for all of them (see prints in the notebook).

I tested and checked the accuracy function many times. I was suspicious of it because my best model gives me around 97% of accuracy for the test set. This is TOO good. 
But still I cannot see an error in my accuracy function and do believe the numbers are relevant. Please see part 4 to understand why accuracy is so high. 


## Notes on Part 4.

*Testing discussion*


The best model 'RNN_nl3_10.pt' has `hp`: 
{
                          "learning_rate": 0.0025,
                          "number_layers": 3,
                          'batch_size' : 10, 
                          'dropout': 0.2,
                          'bidirectional': False,
                          'device' :  f"{device.type}:{device.index}",
                          'hidden_dim' : 64,
                          'emb_dim': 700
                        }.

Smaller lr, bigger emb_dim and almost the same dropout performed much better - stable accuracy growth and loss shrinking. 

The worse model reaches 57% accuracy score in validation. The small dropout in 'RNN_nl2_10' might lead to overfitting, that's what the print statement for loss signals about. 

The best model grows from 92 to 97,96% in accuracy on the val set, and gives 97,81% on the test set. The loss seems to shrink during the training loop. 


## Notes on Part Bonus.

*VERY VERY CONFUSED ABOUT MY SCORES*

Please see my `predict` function. It's imported in run.ipynb from a `predict.py`.
It will randomly choose a sentence from the test test. It will make a prediction on it and give an extensive print statement per token whether:
- it was correct (meaning 100% correct, one prediction was given for this token - crude accuracy as in `calc_accuracy`)
- it was not correct (prediction was fully wrong or it couldn't make a prediction for the token being of any ner_id, e.g. all ner_id predictions got zeroes)
- it was ambiguous - model gave saveral predictions per token and after rounding, several got '1' as a prediction (as in example 2 tensor in part 2). In addition, 'ambiguous' section print  would also tell you whether in those that got labeled as '1' was a correct prediction and whethere the prediction that got maximum score was correct (see example below):

this is a tensor as in example 2 but not rounded (numbers are arbitrary):

[[[0.8999., 0.9919, 0.9919,  ..., 0.9919, 0.9919, 0.], # ner_id 0

[0.5919, 0., 0.,  ..., 0., 0., 0.],                     # ner_id 1

[0.6919, 0., 0.,  ..., 0., 0., 0.7735],                 # ner_id 2

[0.7919, 0., 0.,  ..., 0., 0., 0.],                     # ner_id 3

[0.8919, 0., 0.,  ..., 0., 0., 0.]]                     # ner_id 4


Let's say that ner_id 0 for token #1 is a correct prediction.
We can see that it got the biggest score 0.8999. That is bigger than any other scores for token # 1. The rest being 0.5919 (ner_id 1), 0.6919 (ner_id 2), 0.7919 (ner_id 3), 0.8919 (ner_id 4).

The 'ambiguous' section would tell you: "{'ner_id 0 (None)} got the max score, but labels {1,2,3,4 - error labels} were also assign for {token #1}, which is {ner_id 0 (the correct one)}".
Please note, this logic, which is looking at ambiguous predictions is not included in `calc_accuracy`, meaning, that an example as this would not be considered as correct during testing and validation even when the correct label got the max score.

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
