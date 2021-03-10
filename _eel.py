import eel
import dill
from _utils import clean_text, pad_sequences, pd
from tensorflow.keras.models import load_model
# from _class import *
eel.init('web')

sentiment_analyser = load_model('models/model')
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = dill.load(f)


@eel.expose
def run(data):
    df = pd.DataFrame(data)
    text = ''
    keys = ['text', 'sentences', 'sentiment_text']
    for key in keys:
        if not text:
            try:
                text = df[key]
            except KeyError:
                continue
        break
    if text.empty:
        return {'error': ('Ensure data has any of the following columns', keys)}
    df['PreprocessedText'] = text.apply(clean_text)
    tokenized_text = tokenizer.texts_to_sequences(df['PreprocessedText'])
    padded_text = pad_sequences(tokenized_text, maxlen=140, padding='post')
    df['Predictions'] = sentiment_analyser.predict_classes(padded_text)
    _predictions = df['Predictions'].map({
        1: 'positive',
        0: 'negative'
    })
    df['Mapped Predictions'] = _predictions
    df.to_csv('LatestClassification.csv')
    return df.sample(10).to_dict()


eel.start('index.html')
