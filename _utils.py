from imports import *

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# Preprocess function
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text, re.UNICODE)
    text = text.lower()
    text = [lemmatizer.lemmatize(token) for token in text.split(" ")]
    text = [lemmatizer.lemmatize(token, "v") for token in text]
    text = [word for word in text if word not in stop_words]
    text = " ".join(text)
    return text


# Input Dataset
def prepare_data(path):
    print("=== Inputing Dataset======")
    dataFrame = pd.read_csv(path, error_bad_lines=False, encoding='ISO-8859-1')
    # dataFrame.columns = ["SentimentText", "Sentiment"]
    dataFrame['Sentiment'] = dataFrame['Sentiment'].map({
        'positive': 1,
        'negative': 0
    })
    print("Preprocessing")
    dataFrame['Preprocessed_Text'] = dataFrame['SentimentText'].apply(clean_text)
    sequence_max_len = int(dataFrame.Preprocessed_Text.apply(lambda x: len(x.split(" "))).mean() + 1)

    print("Splititng dataa into train and test sets")
    x_train, y_train, x_test, y_test = train_test_split(
        dataFrame['Preprocessed_Text'],
        dataFrame['Sentiment'],
        test_size=0.3,
        random_state=42
    )

    max_word_feature = 100000
    tokenizer = Tokenizer(max_word_feature)
    tokenizer.fit_on_texts(x_train)
    x_train = tokenizer.texts_to_sequences(x_train)
    x_test = tokenizer.texts_to_sequences(x_test)
    x_train = pad_sequences(x_train, maxlen=sequence_max_len, padding='post')
    x_test = pad_sequences(x_test, maxlen=sequence_max_len, padding='post')
    return x_train, y_train, x_test, y_test, max_word_feature, sequence_max_len
