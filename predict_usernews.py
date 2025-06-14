import nltk
nltk.download('stopwords')
nltk.download('wordnet')

import joblib
import re
import string
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        # Convert to pandas Series if not already
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return X.apply(clean_text)

try:
    pipeline = joblib.load('pipeline_with_linearSVC.joblib')
except FileNotFoundError:
    print("Error: Model file 'pipeline_with_linearSVC.joblib' not found. Please ensure it exists in the current directory.")
    exit(1)

user_text = input("Enter news text to classify as real or fake: ")

if not user_text.strip():
    print("Error: No input provided. Please enter some news text to classify.")
else:
    prediction = pipeline.predict([user_text])
    if prediction == 1:
        print("Prediction: FAKE news")
    else:
        print("Prediction: REAL news")
