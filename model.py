from data_utils import *


class BiLSTM:
    def __init__(self, variant, _dir):
        if variant is 'new':
            self.prediction_data = None
            self.model = None
            self.X = self.Y = None
            self.X_Train = self.X_Test = None
            self.Y_Train = self.Y_Test = None
            self.tokenizer = None
            self.embedding_dimension = 300
            self.embedding_matrix = None
            self.vocab_size = None
            self.maxlen = None
        elif variant is 'load':
            if _dir:
                dependencies = {
                    'precision_m': precision_m,
                    'recall_m': recall_m,
                    'f1_m': f1_m
                }
                self.model = load_model(_dir + '/model.h5', custom_objects=dependencies)
                with open(_dir + '/tokenizer.pkl', 'rb') as f:
                    self.tokenizer = dill.load(f)
                with open(_dir + '/variables.pkl', 'rb') as f:
                    var = dill.load(f)
                if var:
                    self.maxlen = var.maxlen

    @staticmethod
    def preprocess(text):
        punc = string.punctuation.replace('.', '').replace(',', '')
        text = text.lower()
        text = re.sub(r'http\S+', ' ', text)
        text = re.sub(r"i'm", "i am", text)
        text = re.sub(r"he's", "he is", text)
        text = re.sub(r"she's", "she is", text)
        text = re.sub(r"that's", "that is", text)
        text = re.sub(r"where's", "where is", text)
        text = re.sub(r"what's", "what is", text)
        text = re.sub(r"\'ll", ' will', text)
        text = re.sub(r"\'ve", ' have', text)
        text = re.sub(r"\'d", ' would', text)
        text = re.sub(r"won't", "will not", text)
        text = re.sub(r"can't", "cannot", text)
        text = re.sub(r"\'re", ' are', text)
        text = re.sub(r'\(\d+\)', ' ', text)
        text = re.sub(r'pic.\S+', ' ', text)
        text = re.sub(r'@\s\w+', ' ', text)
        text = re.sub(r'#', ' ', text)
        text = re.compile(r'<[^>]+>').sub(' ', text)
        text = re.sub(r'[' + punc + ']', ' ', text)
        text = re.sub(r'RT', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def input_data(self, variant, path):
        f_xtsn = path.split(".")[-1]
        dataFrame = None
        if f_xtsn is 'csv':
            dataFrame = pd.read_csv(path, error_bad_lines=False, encoding='ISO-8859-1')
        elif f_xtsn is 'json':
            dataFrame = pd.read_json(path, lines=True)
        if variant is 'train':
            self.X = dataFrame['text'].astype('str').appy(self.preprocess)
            self.Y = dataFrame['sentiment']
            self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.X, self.Y, test_size=0.25,
                                                                                    random_state=42)
        elif variant is 'predict':
            self.prediction_data = dataFrame['text'].astype('str').appy(self.preprocess)

    def tokenize(self, variant, text):
        start = time()
        if variant:
            if variant is 'train':
                self.tokenizer.fit_on_texts(self.X_Train)
                self.X_Train = self.tokenizer.texts_to_sequences(self.X_Train)
                self.X_Test = self.tokenizer.texts_to_sequences(self.X_Test)
                self.vocab_size = len(self.tokenizer.word_index) + 1
                print("Found {} unique words".format(self.vocab_size))
                print('Tokenization time taken : {}s'.format(time() - start))
                self.maxlen = max(len(x) for x in self.X_Train)
                start = time()
                self.X_Train = pad_sequences(self.X_Train, padding='post', maxlen=self.maxlen)
                self.X_Test = pad_sequences(self.X_Test, padding='post', maxlen=self.maxlen)
                print('Padding Time taken : {}s'.format(time() - start))
            elif variant is 'predict':
                self.prediction_data = self.tokenizer.texts_to_sequences(self.prediction_data)
                self.prediction_data = pad_sequences(self.prediction_data, padding='post', maxlen=self.maxlen)
                print('Time taken : {}s'.format(time() - start))
        if text:
            text = [text]
            text = self.tokenizer.texts_to_sequences(text)
            text = pad_sequences(text, padding='post', maxlen=self.maxlen)
            print('Time taken : {}s'.format(time() - start))
            return text

    def initialize_embedding(self):
        emb = api.load('word2vec-google-news-300')
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dimension))
        for word, index in tqdm(self.tokenizer.word_index.items(),
                                desc="Word2Vec Embedding Vector"):
            try:
                embedding_vector = emb[word]
            except KeyError:
                continue
            if embedding_vector is not None:
                self.embedding_matrix[index] = embedding_vector

    def create_model(self, embedding):
        self.model = Sequential()
        if embedding:
            self.initialize_embedding()
            self.model.add(Embedding(self.vocab_size, self.embedding_dimension, weights=[self.embedding_matrix],
                                     input_length=self.maxlen, trainable=False))
        else:
            self.model.add(Embedding(self.vocab_size, self.embedding_dimension, input_length=self.maxlen))
        self.model.add(
            Bidirectional(LSTM(256, return_sequences=True), input_shape=(self.maxlen, self.embedding_dimension)))
        self.model.add(SeqSelfAttention(attention_activation='sigmoid'))
        # self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy',
                           metrics=['accuracy', precision_m, recall_m, f1_m])
        self.model.summary()

    def train(self):
        es = EarlyStopping(monitor='val_loss', mode='min', patience=2, verbose=1)
        history = self.model.fit(self.X_Train,
                                 self.Y_Train,
                                 epochs=20,
                                 verbose=True,
                                 validation_data=(self.X_Test, self.Y_Test),
                                 batch_size=128,
                                 callbacks=[es])
        _, train_accuracy, train_precision, train_recall, train_fscore = self.model.evaluate(self.X_Train,
                                                                                             self.Y_Train,
                                                                                             verbose=True)
        print("Training Accuracy: {:.4f}".format(train_accuracy))
        print("Training metrics")
        print(f'Precision : {train_precision:{5}} Recall : {train_recall:{5}} F-Score : {train_fscore:{5}}')
        return history, train_accuracy, train_precision, train_recall, train_fscore

    def create_checkpoint(self):
        dte = str(date.today())
        ckpt_dir = '/models/' + dte
        os.makedirs(ckpt_dir)
        self.model.save(ckpt_dir + '/model.h5')
        with open(ckpt_dir + '/tokenizer.pkl', 'wb') as f:
            dill.dump(self.tokenizer, f)
        with open(ckpt_dir + '/variables.pkl', 'wb') as f:
            dill.dump({
                'maxlen': self.maxlen
            }, f)

    def validate(self):
        _, test_accuracy, test_precision, test_recall, test_fscore = self.model.evaluate(self.X_Test,
                                                                                         self.Y_Test,
                                                                                         verbose=True)
        print("Testing Accuracy:  {:.4f}".format(test_accuracy))
        print("Testing metrics")
        print(f'Precision : {test_precision:{5}} Recall : {test_recall:{5}} F-Score : {test_fscore:{5}}')
        return test_accuracy, test_precision, test_recall, test_fscore

    def predict(self, internal, data):
        if internal:
            pred = self.model.predict(self.prediction_data)
        else:
            if isinstance(data, list):
                data = [self.preprocess(x) for x in data]
            if isinstance(data, str):
                data = self.preprocess(data)
                data = [data]
            text = self.tokenizer.texts_to_sequences(data)
            text = pad_sequences(text, padding='post', maxlen=self.maxlen)
            pred = self.model.predict(text)
        return pred
