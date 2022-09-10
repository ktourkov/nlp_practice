

import pandas as pd
import numpy as np

# for handling text
import string
import nltk
import seaborn as sns
import sklearn
import regex as re
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import time
from termcolor import colored
stop_words = set(stopwords.words('english'))

# for plots
import matplotlib as plty
import matplotlib.pyplot as plt
import plotly.offline as py
import plotly.graph_objs as go
import plotly.tools as tls
import plotly.express as px

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score

import tensorflow as tf
# import keras

from tensorflow.keras import layers, callbacks # type: ignore
from sklearn.model_selection import KFold
from keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, LSTM, Dense, SimpleRNN, BatchNormalization  # type: ignore

from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore

def tokenize_text(df, input_col='response_text', output_col="word_list"):
    """
    takes a dataset and name of a column
    then tokenizes the text in the column of that dataset
    """
    df.loc[:, output_col] = df.loc[:, input_col].apply(lambda t: word_tokenize(t))
    return df



def remove_stopwords(list_of_words):
    filtered_list = [w for w in list_of_words if not w.lower() in stop_words]
    return filtered_list

def check_punct(list_of_words):
    """
    look at the tokenized text. if there was any punctuation, it is redundant.
    """
    filtered_list = []
    for word in list_of_words:
        if re.findall("[()!><.,`?']", word):
            pass
        else: filtered_list.append(word)
        
    return filtered_list

def clean(df, input_col='word_list', output_col="cleaned_word_list"):
    
    """
    takes a column of textual data 
    outputs a df with cleaned text attached
    """
    texts = df.loc[:, input_col]
    word_list = []
    
    for text in texts: 
        t = remove_stopwords(text)
        t = check_punct(t)
        word_list.append(t)
        
    df.loc[:, output_col] = word_list
    
    return df


def try_this(t):
    try: word_tokenize(t)
    except: print(t)


def join_tokens(df, tokens_arrays_col):
    return [" ".join(df.loc[i, tokens_arrays_col]) for i in range(len(df))]


def count_words(tokens_arrays):
    """
    gets a dictionary and counts the values
    output: a sorted dict
    note: you can also use a bag of words package to do this
    """
    count_dict = {}
    for array_ in tokens_arrays:
        for word in array_:
            try: count_dict[word] +=1
            except: count_dict[word] = 1

    # sort 
    sorted_count_dict = {k:v for k,v in sorted(count_dict.items(), key=lambda item: item[1], reverse=True)}
    
    return sorted_count_dict

def get_n_key_and_value(n, dict_):

    """
    get the first - most frequent and important -
    words of dictionary 
    """
    keys = [k for (k, v) in dict_.items()][:n]
    values = [v for (k, v) in dict_.items()][:n]

    return keys, values


def convert_tokens_list_to_freq_df(tokens_arrays, n=-1):
    """
    gets the array of tokenized sentences
    output: a sorted dataframe with two cols
    the words and their frequency
    """

    dict_ = count_words(tokens_arrays)
    keys, values = get_n_key_and_value(n, dict_)

    df = pd.DataFrame({'words': keys, 'freq': values})

    return df


def get_tfidf_words_and_array(text_arrays):

    vectorizer = TfidfVectorizer()
    transformed_data = vectorizer.fit_transform(text_arrays).toarray()
    words = vectorizer.get_feature_names_out()
    
    return transformed_data, words

def create_tfidf_df(text_arrays):
    """
    gets the df, converts it into tfidf arrays and words
    then puts them in a dataset
    """

    transformed_data, words = get_tfidf_words_and_array(text_arrays)

    df = pd.DataFrame(data=transformed_data, columns=words).sum().reset_index()

    col_names = ['words', 'tfidf_score_sum']
    default_col_names = df.columns

    # rename whatever the df cols are called to the col_names
    df.rename(columns={default_col_names[i]:col_names[i] for i in range(len(col_names))}, inplace=True)

    return df


def merge(df_1, df_2, on='words'):
    return pd.merge(left=df_1, right=df_2, on=on, how='left')



def to_categorical_tensor(x, num_classes, max_len):
    """
    x: [0, 1, 2, 3]
    output: tensor of one hot encoded text
    with the shape of (max sequence length, number of features) 
    """
    if type(x) != list: a = list(x)
    a = tf.keras.utils.to_categorical(x, num_classes)
    a = tf.constant(a, shape=[max_len, num_classes])
    return a





def evaluate(X, Y, model_, n_epochs=20, n_splits=3, batch_s=32):
    """
    Evaluates each model with cross validation
    """
    history = {}
    kfold = KFold(n_splits)
    splits = kfold.split(X)
    
    early_stopping = callbacks.EarlyStopping(
        min_delta=0.001, # minimium amount of change to count as an improvement
        patience=10, # how many epochs to wait before stopping
        restore_best_weights=True,
    )

    for i, (train_index, val_index) in enumerate(splits):

        print(f'\nfold {i+1}')
        model = model_()

        train_x = X[train_index]
        train_y = Y[train_index]

        val_x = X[val_index]
        val_y = Y[val_index]

        history[f'fold {i}'] = model.fit(
            train_x, train_y, epochs=n_epochs, batch_size=batch_s, verbose=2,
            callbacks = [early_stopping],
             validation_data=(val_x, val_y)).history
        
    mean_val_acc = [np.mean(hist['val_accuracy']) for hist in history.values()]
    return mean_val_acc