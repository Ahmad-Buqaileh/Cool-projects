import pandas as pd

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

import joblib

df = pd.read_csv('text.csv')
nltk.download('stopwords', quiet=True)

model_filename = 'logistic_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'

try:
    classifier = joblib.load(model_filename)
    print("Model loaded successfully.")
    tfidf = joblib.load(vectorizer_filename)
    print("TF-IDF Vectorizer loaded successfully.")

except (FileNotFoundError, OSError):
    Stopwords = stopwords.words('english')
    Stopwords.remove('not')
    ps = PorterStemmer()
    corpus = []

    for text in df['text']:
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        text = [ps.stem(word) for word in text if word not in Stopwords]
        corpus.append(' '.join(text))
    tfidf = TfidfVectorizer(max_features=100000, min_df=2)
    X = tfidf.fit_transform(corpus)
    y = df.iloc[:, -1].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    classifier = LogisticRegression()
    classifier.fit(X_train, y_train)

    joblib.dump(classifier, model_filename)
    joblib.dump(tfidf, vectorizer_filename)
    print("Model and TF-IDF Vectorizer trained and saved successfully.")


def user_input(user_text):
    """Preprocess the input text and return the prediction."""
    user_text = re.sub('[^a-zA-Z]', ' ', user_text)
    user_text = user_text.lower().split()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not')
    PS = PorterStemmer()
    user_text = [PS.stem(word) for word in user_text if word not in set(all_stopwords)]
    user_text = " ".join(user_text)

    text_tfidf = tfidf.transform([user_text])

    Prediction = classifier.predict(text_tfidf)
    return Prediction


input_text = input('Write your text')
prediction = user_input(input_text)
if prediction == 0:
    print('sad')
elif prediction == 1:
    print('joy')
elif prediction == 2:
    print('love')
elif prediction == 3:
    print('anger')
elif prediction == 4:
    print('fear')
else:
    print('surprise')
