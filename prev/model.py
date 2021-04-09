from .data_utils import *


class BiLSTM:
    def __init__(self, variant, _dir):
        if variant is 'new':
            self.prediction_data = None
            self.model = None
            self.X = self.Y = None
            self.X_Train = self.X_Test = None
            self.Y_Train = self.Y_Test = None
            self.tokenizer = None
            self.embedding_dimension = 200
            self.embedding_matrix = None
            self.vocab_size = None
            self.maxlen = None
        elif variant is 'load':
            if _dir:
                self.model = load_model(_dir)
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
        if f_xtsn == 'csv':
            dataFrame = pd.read_csv(path, error_bad_lines=False, encoding='ISO-8859-1')
        elif f_xtsn == 'json':
            dataFrame = pd.read_json(path, lines=True)
        if variant is 'train':
            self.X = dataFrame['SentimentText'].astype('str').apply(self.preprocess)
            self.Y = dataFrame['Sentiment']
            self.X_Train, self.X_Test, self.Y_Train, self.Y_Test = train_test_split(self.X, self.Y, test_size=0.25,
                                                                                    random_state=42)
        elif variant is 'predict':
            self.prediction_data = dataFrame['text'].astype('str').apply(self.preprocess)

    def tokenize(self, variant, text):
        start = time()
        if variant:
            if variant is 'train':
                self.tokenizer = Tokenizer(10000)
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

    def create_model(self):
        metrics = [
            TruePositives(name='tp'),
            TrueNegatives(name='tn'),
            FalsePositives(name='fp'),
            FalseNegatives(name='fn'),
            BinaryAccuracy(name='accuracy'),
            Precision(name='precision'),
            Recall(name='recall')
        ]
        model_input = Input(shape=(self.maxlen, ), dtype='int32')
        x = Embedding(self.vocab_size, self.embedding_dimension, input_length=self.maxlen)(model_input)
        x = Bidirectional(LSTM(128, return_sequences=True), input_shape=(self.maxlen, self.embedding_dimension))(x)
        # noinspection PyArgumentList
        y = Attention(64)
        x = y(x)
        x = Dense(1, activation='sigmoid')(x)
        self.model = Model(model_input, x)
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=metrics)
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
        return history

    def create_checkpoint(self):
        dte = str(date.today())
        ckpt_dir = '/models/' + dte
        os.makedirs(ckpt_dir)
        self.model.save(ckpt_dir)
        with open(ckpt_dir + '/tokenizer.pkl', 'wb') as f:
            dill.dump(self.tokenizer, f)
        with open(ckpt_dir + '/variables.pkl', 'wb') as f:
            dill.dump({'maxlen': self.maxlen}, f)

    def validate(self):
        prediction = self.model.predict_classes(self.X_Test)
        y_pred = (prediction > 0.5)
        report = classification_report(self.Y_Test, y_pred)
        print("Classification Report")
        print(report)
        print("Confusion Matrix")

        def plot_cm(labels, predictions):
            cm = confusion_matrix(labels, predictions)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion matrix (non-normalized))")
            plt.ylabel("Actual label")
            plt.xlabel("Predicted label")

        plot_cm(self.Y_Test, y_pred)
        pass

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

    @staticmethod
    def plot(hist):
        plot_history(hist)
