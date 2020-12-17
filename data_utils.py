import os
import re
import dill
import string
from datetime import date
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm

from keras import backend as k
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras_self_attention import SeqSelfAttention
from keras.preprocessing.sequence import pad_sequences
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Flatten, Dense, Embedding, Input, LSTM, Bidirectional, Dropout

# import gensim.downloader as api
from sklearn.model_selection import train_test_split


def recall_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + k.epsilon()))


api = {}

