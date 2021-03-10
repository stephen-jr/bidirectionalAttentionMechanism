from model import BiLSTM

model = BiLSTM(variant='new', _dir=None)
# model = BiLSTM(variant='load', _dir='models/2020-12-17')
model.input_data(variant='train', path='../data/train.csv')
# model.input_data(variant='predict', path='data/train.csv')
model.tokenize(variant='train', text=None)
# model.tokenize(variant='predict', text=None)
# tokenized_text =  model.tokenize(text='I love Bidirectional LSTM Networks with Self Attention')
# model.create_model(embedding=True)
model.create_model()
history = model.train()
model.plot(hist=history)
model.create_checkpoint()
prediction = model.predict(internal=False, data='I love Bidirectional LSTM Networks')
# prediction = model.predict(internal=True)  # Predict on inputed predict data
