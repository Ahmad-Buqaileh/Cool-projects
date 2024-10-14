import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import joblib

nltk.download('stopwords')


class Model:
    def __init__(self, model_filename='emotions/logistic_model.pkl',
                 vectorizer_filename='emotions/tfidf_vectorizer.pkl'):
        self.model_filename = model_filename
        self.vectorizer_filename = vectorizer_filename
        self.Stopwords = stopwords.words('english')
        self.Stopwords.remove('not')
        self.ps = PorterStemmer()
        self.load_model_and_vectorizer()

    def load_model_and_vectorizer(self):
        """Load model and vectorizer if they exist, otherwise raise an error."""
        try:
            self.classifier = joblib.load(self.model_filename)
            print("Model loaded successfully.")
            self.tfidf = joblib.load(self.vectorizer_filename)
            print("TF-IDF Vectorizer loaded successfully.")
        except (FileNotFoundError, OSError) as e:
            print(f"Error loading model or vectorizer: {e}")
            raise FileNotFoundError("Model or Vectorizer files not found. Ensure the pkl files are present.")

    def preprocess_text(self, text):
        """Preprocess text by removing non-alphabetic characters, stemming, and removing stopwords."""
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        return ' '.join([self.ps.stem(word) for word in text if word not in self.Stopwords])

    def predict(self, user_text):
        """Preprocess input text and predict sentiment using the trained model."""
        processed_text = self.preprocess_text(user_text)
        text_tfidf = self.tfidf.transform([processed_text])
        prediction = self.classifier.predict(text_tfidf)
        return prediction

    def get_sentiment_label(self, prediction):
        """Return the corresponding sentiment label based on the prediction."""
        sentiment_dict = {
            0: 'sad',
            1: 'joy',
            2: 'love',
            3: 'anger',
            4: 'fear',
            5: 'surprise'
        }
        return sentiment_dict.get(prediction[0])
