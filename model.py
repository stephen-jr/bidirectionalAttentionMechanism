import pandas as pd
import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Embedding, Input, LSTM, Bidirectional, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint



class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def BiLstm(maxlen, max_features, embed_size, embedding_matrix):
    inp = Input(shape=(maxlen, ))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix],
                  trainable=False)(inp)
    x = Bidirectional(LSTM(300, return_sequences=True, dropout=0.25,
                           recurrent_dropout=0.25))(x)
    x = Attention(maxlen)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.25)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)

    return model
    
    
def make_df(train_path, test_path, max_features, maxlen, list_classes):
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train = train.sample(frac=1)

    list_sentences_train = train["comment_text"].fillna("unknown").values
    y = train[list_classes].values
    list_sentences_test = test["comment_text"].fillna("unknown").values

    tokenizer = Tokenizer(num_words=max_features)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
    X_train = pad_sequences(list_tokenized_train, maxlen=maxlen)
    X_test =pad_sequences(list_tokenized_test, maxlen=maxlen)

    word_index = tokenizer.word_index

    return X_t, X_te, y, word_index
    
max_features = 100000
maxlen = 150
embed_size = 300
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

xtr, xte, y, word_index = make_df("../input/jigsaw-toxic-comment-classification-challenge/train.csv", 
                                  "../input/jigsaw-toxic-comment-classification-challenge/test.csv", 
                                  max_features, 
                                  maxlen, 
                                  list_classes)
embedding_vector = make_glovevec("../input/glove840b300dtxt/glove.840B.300d.txt", 
                                 max_features, 
                                 embed_size, 
                                 word_index)

model = BidLstm(maxlen, max_features, embed_size, embedding_vector)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
file_path = ".model.hdf5"
ckpt = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor="val_loss", mode="min", patience=1)
model.fit(xtr, y, batch_size=256, epochs=15, validation_split=0.1, callbacks=[ckpt, early])

model.load_weights(file_path)
y_test = model.predict(xte)