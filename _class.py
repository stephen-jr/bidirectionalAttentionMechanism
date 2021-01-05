from imports import *


class Attention(keras.Model):
    def __init__(self, units):
        super(Attention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        hidden_with_time_axis = tf.expand_dims(hidden, 1)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = tf.nn.softmax(self.V(score), axis=1)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights


class NN:
    def __init__(self, maxlen, max_features, embedding_size, cell_size):
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
        self.model = None

    def create(self):
        print("=== Generating Attention Based RNN ===")
        es = EarlyStopping(monitor='val_acc', patience=1, verbose=1)
        sequence_input = Input(shape=(self.MAX_LEN,), dtype="int32")
        embedded_sequences = Embedding(self.MAX_FEATURES, self.EMBED_SIZE)(sequence_input)
        lstm = Bidirectional(
            LSTM(
                self.RNN_CELL_SIZE,
                return_sequences=True
            ),
            name="bi_lstm"
        )(embedded_sequences)
        (
            lstm,
            forward_h,
            forward_c,
            backward_h,
            backward_c
        ) = Bidirectional(
            LSTM(
                self.RNN_CELL_SIZE,
                return_sequences=True,
                return_state=True
            ),
            name="bi_lstm_1"
        )(lstm)
        state_h = Concatenate()([forward_h, backward_h])
        # state_c = Concatenate()([forward_c, backward_c])
        context_vector, attention_weights = Attention(10)(lstm, state_h)
        dense1 = Dense(20, activation="relu")(context_vector)
        dropout = Dropout(0.05)(dense1)
        output = Dense(1, activation="sigmoid")(dropout)
        model = keras.Model(inputs=sequence_input, outputs=output)
        print(model.summary())
        keras.utils.plot_model(model, show_shapes=True, dpi=90)
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=self.METRICS)
        self.model = model

    def train(self, x_train, y_train, batch_size, epochs):
        try:
            history = self.model.fit(
                x_train,
                y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_split=0.2)
            return history
        except TypeError as e:
            print("Pass in the necessary parameter(s)", e)

    def validate(self, x_test, y_test):
        prediction = self.model.predict(x_test)
        y_pred = (prediction > 0.5)
        report = classification_report(y_test, y_pred)
        print("Classification Report")
        print(report)
        print("Confusion Matix")

        def plot_cm(labels, predictions):
            cm = confusion_matrix(labels, predictions)
            plt.figure(figsize=(5, 5))
            sns.heatmap(cm, annot=True, fmt="d")
            plt.title("Confusion matrix (non-normalized))")
            plt.ylabel("Actual label")
            plt.xlabel("Predicted label")

        plot_cm(y_test, y_pred)
        return report

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
