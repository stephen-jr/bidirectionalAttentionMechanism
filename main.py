from model import BiLSTM

model = BiLSTM(variant='new', _dir=None)
# model = BiLSTM(variant='load', _dir='models/2020-12-17')
model.input_data(variant='train', path='data/train.csv')
# model.input_data(variant='predict', path='data/train.csv')
model.tokenize(variant='train', text=None)
# model.tokenize(variant='predict', text=None)
# tokenized_text =  model.tokenize(text='I love Bidirectional LSTM Networks with Self Attention')
model.create_model(embedding=True)
# model.create_model(embedding=False)
history, train_accuracy, train_precision, train_recall, train_fscore = model.train()
model.create_checkpoint()
test_accuracy, test_precision, test_recall, test_fscore = model.validate()
prediction = model.predict(data='I love Bidirctional LSTM Networks')
# prediction = model.predict(internal=True)  # Predict on inputed predict data
