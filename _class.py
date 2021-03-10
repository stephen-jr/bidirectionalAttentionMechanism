from imports import *


class NN:
    def __init__(self, maxlen, max_features, embedding_size, cell_size):
        try:
            assert maxlen, max_features is not None
            assert embedding_size, cell_size is not None
            self.MAX_LEN = maxlen
            self.MAX_FEATURES = max_features
            self.EMBED_SIZE = embedding_size
            self.RNN_CELL_SIZE = cell_size
            self.METRICS = [
                keras.metrics.BinaryAccuracy(name='accuracy'),
                keras.metrics.Precision(name='precision'),
                keras.metrics.Recall(name='recall'),
                keras.metrics.AUC(name='auc'),
            ]
        except AssertionError:
            pass
        finally:
            self.model = None

    def create(self):
        print("=== Generating Attention Based RNN ===")
        model = Sequential(Embedding(self.MAX_FEATURES, self.EMBED_SIZE, input_length=self.MAX_LEN))
        model.add(Bidirectional(LSTM(self.RNN_CELL_SIZE, return_sequences=True)))
        model.add(Attention())
        model.add(Dropout(0.5))
        model.add(Dense(1, activation="sigmoid"))
        print(model.summary())
        keras.utils.plot_model(model, show_shapes=True, dpi=90)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=self.METRICS)
        self.model = model

    def train(self, x_train, y_train, batch_size, epochs):
        es = EarlyStopping(monitor='val_acc', patience=1, verbose=1)
        try:
            history = self.model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2,
                callbacks=[es]
            )
            return history
        except TypeError as e:
            print("Pass in the necessary parameter(s)", e)

    def validate(self, x_test, y_test):
        prediction = self.model.predict(x_test)
        y_pred = (prediction > 0.5)
        report = classification_report(y_test, y_pred)
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

        plot_cm(y_test, y_pred)
        # return report

    def save_model(self, tokenizer=None):
        if self.model is not None:
            tokenizer_path = 'models/tokenizer.pkl'
            if os.path.exists(tokenizer_path):
                os.remove(tokenizer_path)
            if tokenizer is not None:
                with open(tokenizer_path, 'wb') as f:
                    dill.dumps(tokenizer, tokenizer_path)
            self.model.save('models/bidirectionalAttentionModel')
        else:
            raise AttributeError('Model is not Initialized')

    def ld_model(self):
        tokenizer_path = 'models/tokenizer.pkl'
        self.model = load_model('models/bidirectionalAttentionModel')
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = dill.load(f)
            return tokenizer
        else:
            raise FileNotFoundError("Tokenizer file does not exist")

    @staticmethod
    def plot_metrics(history):
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
