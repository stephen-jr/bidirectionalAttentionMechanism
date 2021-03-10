import os
import re
import sys
import dill

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.models import load_model
from keras.layers import Bidirectional
from keras.models import Model, Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import Concatenate, Dense, Input, LSTM, Embedding, Dropout

from sklearn import svm, datasets
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from attention import Attention

# nltk.download('stopwords')
# nltk.download('wordnet')
