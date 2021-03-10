import os
import re
import dill
import string
import numpy as np
import pandas as pd
from time import time
from tqdm import tqdm
from datetime import date
from matplotlib import pyplot as plt
import matplotlib as mpl
from attention import Attention
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, TruePositives, TrueNegatives, FalsePositives, \
    FalseNegatives
from tensorflow.keras.layers import Flatten, Dense, Embedding, Flatten, LSTM, Bidirectional, Dropout, Input

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


def plot_history(history):
    print("Plotting Training Comparison Metrics Metrics")
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mpl.rcParams["figure.figsize"] = (12, 18)
    metrics = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "auc",
    ]
    for n, metric in enumerate(metrics):
        name = metric.replace("_", " ").capitalize()
        plt.subplot(5, 2, n + 1)
        plt.plot(
            history.epoch,
            history.history[metric],
            color=colors[0],
            label="Train",
        )
        plt.plot(
            history.epoch,
            history.history["val_" + metric],
            color=colors[1],
            linestyle="--",
            label="Validation",
        )
        plt.xlabel("Epoch")
        plt.ylabel(name)
        if metric == "loss":
            plt.ylim([0, plt.ylim()[1] * 1.2])
        elif metric == "accuracy":
            plt.ylim([0.4, 1])
        elif metric == "precision":
            plt.ylim([0, 1])
        elif metric == "recall":
            plt.ylim([0.4, 1])
        else:
            plt.ylim([0, 1])
        plt.legend()

