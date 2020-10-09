#basics
import random
import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt 
from venn import venn 
import nltk
from nltk import TreebankWordTokenizer
import glob
from collections import Counter
import xml.etree.ElementTree as ET
import pickle
import os.path

class DataLoaderBase:

    #### DO NOT CHANGE ANYTHING IN THIS CLASS ### !!!!

    def __init__(self, data_dir:str, device=None):
        self._parse_data(data_dir)
        assert list(self.data_df.columns) == [
                                                "sentence_id",
                                                "token_id",
                                                "char_start_id",
                                                "char_end_id",
                                                "split"
                                                ]

        assert list(self.ner_df.columns) == [
                                                "sentence_id",
                                                "ner_id",
                                                "char_start_id",
                                                "char_end_id",
                                                ]
        self.device = device
        

    def get_random_sample(self):
        # DO NOT TOUCH THIS
        # simply picks a random sample from the dataset, labels and formats it.
        # Meant to be used as a naive check to see if the data looks ok
        sentence_id = random.choice(list(self.data_df["sentence_id"].unique()))
        sample_ners = self.ner_df[self.ner_df["sentence_id"]==sentence_id]
        sample_tokens = self.data_df[self.data_df["sentence_id"]==sentence_id]

        decode_word = lambda x: self.id2word[x]
        sample_tokens["token"] = sample_tokens.loc[:,"token_id"].apply(decode_word)

        sample = ""
        for _, t_row in sample_tokens.iterrows():

            is_ner = False
            for _, l_row in sample_ners.iterrows():
                 if t_row["char_start_id"] >= l_row["char_start_id"] and t_row["char_start_id"] <= l_row["char_end_id"]:
                    sample += f'{self.id2ner[l_row["ner_id"]].upper()}:{t_row["token"]} '
                    is_ner = True
            
            if not is_ner:
                sample += t_row["token"] + " "

        return sample.rstrip()

class DataLoader(DataLoaderBase):
    def __init__(self, data_dir:str, device=None):
        super().__init__(data_dir=data_dir, device=device)

        self.max_sample_length = self.discover_max_sample_len()

    @classmethod
    def create_tokens_offs(self, text):
        text_offsets = TreebankWordTokenizer().span_tokenize(text)
        text_tokens = TreebankWordTokenizer().tokenize(text)
        tokens_and_offs = list(zip(text_tokens, text_offsets))

        return tokens_and_offs
    
    def create_data_df(self, root_dir):
        data = {
            'sentence_id': [],
            'token_id': [],
            'char_start_id' : [],
            'char_end_id': [],
            'split': [],
        }

        id2word = [None]

        def add_row(sentence_id, token_id, char_start_id, char_end_id, split):
            data['sentence_id'].append(sentence_id)
            data['token_id'].append(token_id)
            data['char_start_id'].append(int(char_start_id))
            data['char_end_id'].append(int(char_end_id)-1)
            data['split'].append(split)

        i = 0
        
        for filename in glob.iglob(root_dir + "**/**", recursive=True):
            if '.xml' not in filename:
                continue
            filename_split = [path.upper() for path in filename.split('/')]
            split = None
            for path in filename_split:
                if path == 'TRAIN' or path == 'TEST':
                    split = path
                    break
            if split is None:
                raise Exception('Data folder is not structured in Train and Test folders')

            tree = ET.parse(filename)
            root = tree.getroot()

            for sentence in root:
                sentence_id = sentence.attrib['id']
                sentence_text = sentence.attrib['text']
                tokens = DataLoader.create_tokens_offs(sentence_text)
                for t in tokens:
                    if t[0].lower() not in id2word:
                        id2word.append(t[0].lower())
                    char_start_id = t[1][0]
                    char_end_id = t[1][1]

                    token_id = id2word.index(t[0].lower())

                    add_row(sentence_id, token_id, char_start_id, char_end_id, split)
                    
                i += 1

                if i % 1000 == 0:
                    print(f'Finished with {i} sentences')
        
        print("Ready to extract data")

        df = pd.DataFrame.from_dict(data)
        
        print("Doing split")

        test_df = df[df['split']=="TEST"]
        train_and_val_df = df[df['split'] == 'TRAIN']

        sent_unique = train_and_val_df['sentence_id'].unique()

        # the bolow lines make sure that one sentence is not 
        # split beetween val and train dataframes
        distribution_count = round(len(sent_unique)*0.2)
        val_sent_ids = sent_unique[:distribution_count]
        pd.options.mode.chained_assignment = None
        train_and_val_df.loc[train_and_val_df['sentence_id'].isin(val_sent_ids), 'split'] = 'VAL'
        result_df = pd.concat([test_df, train_and_val_df])

        print("Split data in train, test, val is done.")
        ids_train = list(result_df.loc[result_df['split'] == 'TRAIN', 'token_id'].unique()) # list of all ids 
        vocab = [None] + [id2word[i] for i in ids_train]
        # print(len(ids_train))
        return result_df, id2word, vocab

    @classmethod
    def split_offset(self, offset):
        offsets = []
        for offset_part in str(offset).split(';'):
            offset_parts_split = offset_part.split('-')
            offsets.append((int(offset_parts_split[0]), int(offset_parts_split[1])))
        return offsets

    def create_ner_df(self, root_dir):
        data = {
            'sentence_id':[],
            'ner_id': [],
            'char_start_id': [],
            'char_end_id': [],
        }

        id2ner = [None, 'group', 'drug_n', 'drug', 'brand']

        def add_row(sentence_id, ner_id, char_start_id, char_end_id):
            data['sentence_id'].append(sentence_id)     
            data['ner_id'].append(ner_id)
            data['char_start_id'].append(char_start_id)
            data['char_end_id'].append(char_end_id)

        for filename in glob.iglob(root_dir+ '**/**', recursive=True):
            if '.xml' not in filename:
                continue
                
            tree = ET.parse(filename)
            root = tree.getroot()
            for sentence in root:
                sentence_id = sentence.attrib['id']

                for entity in sentence.findall("./entity"):
                    char_offset_text = entity.attrib['charOffset']
                    ner_type = entity.attrib['type']
                    char_offsets = DataLoader.split_offset(char_offset_text)

                    ner_id = id2ner.index(ner_type)
                    for (char_start_id, char_end_id) in char_offsets:
                        add_row(sentence_id, ner_id, char_start_id, char_end_id)

        ner_df = pd.DataFrame.from_dict(data)

        return ner_df, id2ner

    def _parse_data(self, data_dir):
        # Should parse data in the data_dir, create two dataframes with the format specified in
        # __init__(), and set all the variables so that run.ipynb run as it is.
        #
        # NOTE! I strongly suggest that you create multiple functions for taking care
        # of the parsing needed here. Avoid create a huge block of code here and try instead to 
        # identify the seperate functions needed.
        if not os.path.exists(data_dir):
            raise Exception("Data dir does not exist")

        REPICKLE = False
        PICKLE_DATA_FILE = 'data.pickle'

        if REPICKLE == True or not os.path.isfile(PICKLE_DATA_FILE):
            (data_df, id2word, vocab) = self.create_data_df(data_dir)
            (ner_df, id2ner) = self.create_ner_df(data_dir)
            df_files = open(PICKLE_DATA_FILE, 'wb')
            pickle.dump((data_df, id2word, vocab, ner_df, id2ner), df_files)
            df_files.close()
        else:
            df_files = open(PICKLE_DATA_FILE, 'rb')
            data_df, id2word, vocab, ner_df, id2ner = pickle.load(df_files)

        self.data_df = data_df
        self.ner_df = ner_df
        self.id2word = id2word
        self.vocab = vocab
        self.id2ner = id2ner

    def discover_max_sample_len(self):
        sents = self.data_df[['sentence_id']]
        return int(sents['sentence_id'].value_counts().max())

    def extract_tensor_data(self, token_df, ners_df, max_tensor_length):
        merged_df = token_df.merge(ners_df, how= 'left', left_on=['sentence_id', 'char_start_id', 'char_end_id'],\
            right_on = ['sentence_id', 'char_start_id', 'char_end_id'])
        
        # for non-ner tokens
        merged_df['ner_id'] = merged_df['ner_id'].fillna(0)
        # to ensure the correct order of tokens in sent
        merged_df = merged_df.sort_values(by=['sentence_id', 'char_start_id'])

        grouped_by_sentence = merged_df.groupby(['sentence_id'])

        y_tensor = []

        for _, columns in grouped_by_sentence:
            ner_ids = [int(v) for v in list(columns['ner_id'])]
            if len(ner_ids) > max_tensor_length:
                ner_ids = ner_ids[:max_tensor_length]
            else:
                ner_ids += [0] * (max_tensor_length - len(ner_ids))
            y_tensor.append(ner_ids)
        y_tensor = np.asarray(y_tensor)

        return y_tensor 
    
    def split_ner(self):
        train_df = self.data_df[self.data_df['split'] == 'TRAIN']
        val_df = self.data_df[self.data_df['split'] == 'VAL']
        test_df = self.data_df[self.data_df['split'] == 'TEST']

        docs_train = list(train_df['sentence_id'].unique())
        docs_val = list(val_df['sentence_id'].unique())
        docs_test = list(test_df['sentence_id'].unique())

        ner_train_df = self.ner_df[self.ner_df['sentence_id'].isin(docs_train)]
        ner_val_df = self.ner_df[self.ner_df['sentence_id'].isin(docs_val)]
        ner_test_df = self.ner_df[self.ner_df['sentence_id'].isin(docs_test)]
        
        return ner_train_df, ner_val_df, ner_test_df

    def get_y(self):
        # Should return a tensor containing the ner labels for all samples in each split.
        # the tensors should have the following following dimensions:
        # (NUMBER_SAMPLES, MAX_SAMPLE_LENGTH)
        # NOTE! the labels for each split should be on the GPU
        REPICKLE = False
        PICKLE_DATA_FILE = 'split_ner.pickle'

        if REPICKLE == True or not os.path.isfile(PICKLE_DATA_FILE):
            ner_train_df, ner_val_df, ner_test_df = self.split_ner()
            
            pickle_file = open(PICKLE_DATA_FILE, 'wb')
            pickle.dump((ner_train_df, ner_val_df, ner_test_df), pickle_file)
            pickle_file.close()
        else:
            pickle_file = open(PICKLE_DATA_FILE, 'rb')
            ner_train_df, ner_val_df, ner_test_df = pickle.load(pickle_file)

        train_df = self.data_df[self.data_df['split'] == 'TRAIN']
        val_df = self.data_df[self.data_df['split'] == 'VAL']
        test_df = self.data_df[self.data_df['split'] == 'TEST']

        y_train = self.extract_tensor_data(train_df, ner_train_df, self.max_sample_length)
        y_val = self.extract_tensor_data(val_df, ner_val_df, self.max_sample_length)
        y_test = self.extract_tensor_data(test_df, ner_test_df, self.max_sample_length)
        
        y_tensor_train = torch.from_numpy(y_train).to(self.device)
        y_tensor_val = torch.from_numpy(y_val).to(self.device)
        y_tensor_test = torch.from_numpy(y_test).to(self.device)

        return y_tensor_train, y_tensor_val, y_tensor_test

    def plot_split_ner_distribution(self):
        train, val, test = self.get_y()
        train = train.cpu().detach().numpy()
        val = val.cpu().detach().numpy()
        test = test.cpu().detach().numpy()
        # should plot a histogram displaying ner label counts for each split
        def prepare_data(tensor_data):
            unique, counts = np.unique(tensor_data, return_counts=True)
            result = dict(zip(unique, counts))
            # disregard 0 since it's not ner id
            if 0 in result:
                result.pop(0)
            return result

        names = ['TRAIN', 'VAL', 'TEST']
        train_count = prepare_data(train)
        val_count = prepare_data(val)
        test_count = prepare_data(test)

        all_counts = []
        for i in range(1,5):
            all_counts.append([train_count[i], val_count[i], test_count[i]])
        
        fig, ax = plt.subplots(ncols=1, nrows=1)
        x = np.arange(len(names))
        width = 0.10

        for i, values in enumerate(all_counts):
            ax.bar(x+(i*width), values, width=-width, label=f'{i+1}', align='edge')

        ax.set_ylabel("counts")
        ax.set_title(' NER LABELS DISTRIBUTION PER SPLIT ')
        ax.set_xticks(x)
        ax.set_xticklabels(names)
        ax.legend()
        fig.tight_layout()
        # plt.savefig('ner-distr-per-plit.png')
        plt.show()

    def plot_sample_length_distribution(self):
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of sample lengths in number tokens
        train_df = self.data_df[self.data_df['split'] == 'TRAIN']
        test_df = self.data_df[self.data_df['split'] == 'TEST']
        val_df = self.data_df[self.data_df['split'] == 'VAL']

        sents_train = train_df[['sentence_id']]
        sents_test = test_df[['sentence_id']]
        sents_val = val_df[['sentence_id']]

        def get_distribution(df):
            v = df['sentence_id'].values
            counts = Counter(v).most_common()
            lengh_list = [c[1] for c in counts]
            return lengh_list[::-1]

        len_distrib_train = get_distribution(sents_train)
        len_distrib_test = get_distribution(sents_test)
        len_distrib_val = get_distribution(sents_val)

        plt.hist(len_distrib_train, bins = 130, label = "TRAIN", color = 'g')
        plt.hist(len_distrib_test, bins = 130, label = 'TEST', color = 'r')
        plt.hist(len_distrib_val, bins = 130, label = "VAL", color= 'y')

        plt.ylabel('sentences in a split')
        plt.xlabel('length of a sentence')

        plt.title('SAMPLE LENGTH DISTRIBUTION BETWEEN SPLITS') 
        plt.legend()
        # plt.savefig('sample-len.png')
        plt.show()

    def plot_ner_per_sample_distribution(self):        
        # FOR BONUS PART!!
        # Should plot a histogram displaying the distribution of number of NERs in sentences
        # e.g. how many sentences has 1 ner, 2 ner and so on
        train, val, test = self.get_y()
        train = train.cpu().detach().numpy()
        val = val.cpu().detach().numpy()
        test = test.cpu().detach().numpy() 
        
        def count_ners(arr): 
            ners_in_sent = []
            for sent in arr:
                counts = Counter(sent)
            # returns something like Counter({0: 152, 3: 6, 4: 3, 1: 3, 2: 1})
            # we can always disregard 0, that each sent apart from one has
                if 0 in counts:
                    counts.pop(0)
                ners_in_sent.append(len(counts)) 
            return sorted(ners_in_sent)
        
        ners_train = count_ners(train)
        ners_val = count_ners(val)
        ners_test = count_ners(test)

        plt.hist(ners_train, bins = 'auto', histtype='bar', range=(0,5), align='mid',  label="Train", color='g')
        plt.hist(ners_test, bins = 'auto', histtype='bar', range=(0,5), align='mid',  label="Test", color='y')
        plt.hist(ners_val, bins = 'auto', histtype='bar', range=(0,5), align='mid',  label="Val", color='r')

        plt.ylabel('sentences with ner')   
        plt.xlabel('number of ner-labels in a sentence')
        plt.title('NER DISTRIBUTION OVER SENTENCES')
        plt.legend()
        # plt.savefig('ner-distr.png')
        plt.show()

    def plot_ner_cooccurence_venndiagram(self):
        # FOR BONUS PART!!
        # Should plot a ven-diagram displaying how the ner labels co-occur
        venn_data = {}

        for ner_id in self.ner_df['ner_id'].unique():
            ner_entries = self.ner_df[self.ner_df['ner_id'] == ner_id]
            sentence_ids = ner_entries['sentence_id'].unique()

            venn_data[self.id2ner[ner_id]] =set(sentence_ids)

        plot = venn(venn_data)
        plot.set_title('NER DISTRIBUTION BETWEEN SENTENCES')
        fig = plot.get_figure()
        # fig.savefig('venn.png')
        fig.show()
