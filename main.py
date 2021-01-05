from _class import *
from _utils import prepare_data


def main():
    DATAPATH = 'data/train.csv'
    X_TRAIN, Y_TRAIN, X_TEST, Y_TEST, MAX_WORD_FEATURE, SEQUENCE_MAX_LEN = prepare_data(DATAPATH)
    model = NN(SEQUENCE_MAX_LEN, MAX_WORD_FEATURE, 100, 64)
    model.create()
    train_history = model.train(X_TRAIN, Y_TRAIN, 128, epochs=10)
    validation_report = model.validate(X_TEST, Y_TEST)
    model.plot_metrics(train_history)


if __name__ == '__main__':
    main()
