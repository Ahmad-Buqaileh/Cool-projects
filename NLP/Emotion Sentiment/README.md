# Project Overview
### This project aims to build an emotion detection system that classifies text into different emotion categories. By utilizing machine learning techniques, the system will analyze input text and predict one of several predefined emotions, such as joy, sadness, love, anger, fear, or surprise. The application is intended for real-time sentiment analysis, enabling users to input text and receive immediate feedback on the detected emotion.
# Since the DATA is large for github, you can download it here :  https://www.kaggle.com/datasets/nelgiriyewithana/emotions
# In this project, the main task is to create an NLP model. However, I have also created a simple UI for it and connected it to a SQL database; these tasks are optional.
# First, try doing it on your own. If you struggle with something, you can find the steps outlined below.

## Import necessary libraries
```bash
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
```
## Load CSV file containing text data
```bash
df = pd.read_csv('csv_filename') # your csv file here
```
## Download stopwords from nltk (if not already downloaded)
```bash
nltk.download('stopwords', quiet=True)
```
## Filenames to store the trained model and vectorizer
```bash
model_filename = 'logistic_model.pkl'
vectorizer_filename = 'tfidf_vectorizer.pkl'
```
## Attempt to load the model and TF-IDF vectorizer if they were already saved
```bash
try:
    classifier = joblib.load(model_filename)
    print("Model loaded successfully.")
    tfidf = joblib.load(vectorizer_filename)
    print("TF-IDF Vectorizer loaded successfully.")
```
#### If model or vectorizer are not found, preprocess data and train the model
```bash
except (FileNotFoundError, OSError):
```
## Creating model and training it
#### Remove common stopwords from nltk, but keep the word 'not' (important for sentiment analysis)
```bash
    Stopwords = stopwords.words('english')
    Stopwords.remove('not')
```
#### Initialize the PorterStemmer for word stemming
```
    ps = PorterStemmer()
    corpus = []
```
#### Preprocess each text in the dataset
```bash
    for text in df['text']:
       # Remove non-alphabetical characters and convert text to lowercase
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        # Apply stemming and remove stopwords
        text = [ps.stem(word) for word in text if word not in Stopwords]
        corpus.append(' '.join(text))
```
## Create a TF-IDF vectorizer with specified parameters
```bash
    tfidf = TfidfVectorizer(max_features=100000, min_df=2)
    X = tfidf.fit_transform(corpus)  # Fit and transform the preprocessed text data
    y = df.iloc[:, -1].values  # Target labels for classification
 ```
## Split the dataset into training and testing sets
```bash
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
## Initialize and train a Logistic Regression model
```bash
    lassifier = LogisticRegression()
    classifier.fit(X_train, y_train)
```
## Save the trained model and vectorizer for future use
```bash
    joblib.dump(classifier, model_filename)
    joblib.dump(tfidf, vectorizer_filename)
    print("Model and TF-IDF Vectorizer trained and saved successfully.")
```
## Function to preprocess input text and return the predicted sentiment
```bash
def user_input(user_text):
    """Preprocess the input text and return the prediction."""
    user_text = re.sub('[^a-zA-Z]', ' ', user_text)  # Clean text by removing non-alphabetic characters such as , and ' 
    user_text = user_text.lower().split()  # convert to lowercase and split into words
    all_stopwords = stopwords.words('english')  
    all_stopwords.remove('not')  # remove 'not' from stopwords
    PS = PorterStemmer() 
    user_text = [PS.stem(word) for word in user_text if word not in set(all_stopwords)]
    user_text = " ".join(user_text)  

    text_tfidf = tfidf.transform([user_text])  

    Prediction = classifier.predict(text_tfidf) 
    return Prediction
```
## Take user input and provide a prediction on the sentiment. (you can do this part in many ways (0_0))
```bash
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
```
# We've finished the model; now for the extra tasks:                                                          1- A class for the UI.
## Import necessary libraries
```bash
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import joblib
```
## Download stopword (in case something happens :D )
```bash
nltk.download('stopwords')
```
## Model class to handle loading, preprocessing, and predicting sentiment
```bash
class Model:
    def __init__(self, model_filename='path_to_your_pkl_file',
                 vectorizer_filename='path_to_your_pkl_file'):
        self.model_filename = model_filename
        self.vectorizer_filename = vectorizer_filename
        self.Stopwords = stopwords.words('english')
        self.Stopwords.remove('not')
        self.ps = PorterStemmer()
        self.load_model_and_vectorizer()
```
## Load the trained model and vectorizer from disk
```bash
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
```
## Preprocess the input text by cleaning and stemming it                                        
```bash
   def preprocess_text(self, text):
        """Preprocess text by removing non-alphabetic characters, stemming, and removing stopwords."""
        text = re.sub('[^a-zA-Z]', ' ', text)
        text = text.lower().split()
        return ' '.join([self.ps.stem(word) for word in text if word not in self.Stopwords])
```
## Predict sentiment from input text
```bash
def predict(self, user_text):
        """Preprocess input text and predict sentiment using the trained model."""
        processed_text = self.preprocess_text(user_text)
        text_tfidf = self.tfidf.transform([processed_text])
        prediction = self.classifier.predict(text_tfidf)
        return prediction
```
## Get the sentiment label based on the prediction
```bash
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
```
# Now let's create the UI
## Import necessary libraries
```bash
import streamlit as st
from model import Model
import psycopg2
```
## Function to insert user input and the predicted result into the database ( I used PostgreSQL)
```bash
def insert_output(User_Input, Result):
    cur.execute('INSERT INTO sentemint(user_text,label) VALUES (%s,%s)', (user_input, Result))
    connection.commit() # I named my table sentemint, with column names user_text and label
```
## Connect to PostgreSQL database
```bash
connection = psycopg2.connect(host='localhost',
                              database='database_name_here',
                              user='username_here',
                              password='your_password_here')
cur = connection.cursor()
connection.commit()
```
## Initialize the model class
```bash
trained_model = Model()
```
## Streamlit app title and layout
```bash
st.title("Emotion Detection")
st.markdown("<br><br>", unsafe_allow_html=True)
```
## Button to trigger emotion analysis
```bash
if st.button('Analyze'):
    if len(user_input) > 24:  # ensure that user input has more than 24 characters
        if user_input:
            # predict sentiment using the model and display the result
            prediction = trained_model.predict(user_input)
            output = trained_model.get_sentiment_label(prediction)
            st.write(f"## Emotion: {output}")
            # save the user input and prediction result in the database
            insert_output(user_input, output)
        else:
            st.markdown("## Please enter some text")
    else:
        st.markdown("## Please enter more than 24 characters")
```
# Now lets create the database. (again I'm using PostgreSQL)
## Open the SQL shell and create the DB (DataBase)
![image](https://github.com/user-attachments/assets/ad913e0a-e6b1-49e9-9a8a-49516f6d67f1)
## Connect to the newly created DB. ( you can see all of your databases by running \l )
![image](https://github.com/user-attachments/assets/ec32cf62-54b5-451a-8dc5-89ac38f2426f)
## Create a new table with column names user_text and label
![image](https://github.com/user-attachments/assets/2c022a11-214a-4351-8a8f-e53181e4a193)
## To see values  
![image](https://github.com/user-attachments/assets/dcf56ee7-171d-4692-9b88-3116a4c48314)
# At the end everything should look like this 
![image](https://github.com/user-attachments/assets/7fddebac-6d42-4715-9469-47b24c565965)
![image](https://github.com/user-attachments/assets/95ae5826-cae7-4c1b-baaf-92813ac94883)


