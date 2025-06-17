# backendd.py (updated)
import pandas as pd
import joblib
import re
import string
import requests
import os
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from openai import OpenAI
from dotenv import load_dotenv

# Initialize environment
load_dotenv()
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# ============== TEXT CLEANING ==============
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

class TextCleaner(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        if not isinstance(X, pd.Series):
            X = pd.Series(X)
        return X.apply(clean_text)

# ============== MODEL PIPELINE ==============
def load_pipeline():
    return joblib.load('pipeline_with_linearSVC.joblib')  # Move loading into a function

# ============== API INTEGRATION ==============
client = OpenAI(api_key=os.getenv("OPEN_AI_API_KEY"))
FACT_CHECK_URL = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

def extract_claims_with_openai(text):
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": f"Extract factual claims as bullet points: {text}"}]
    )
    claims = response.choices[0].message.content.split("\n")
    return [claim.strip("- ") for claim in claims if claim.strip()]

def google_fact_check(claim):
    params = {
        "query": claim,
        "languageCode": "en-US",
        "pageSize": 3,
        "key": os.getenv("FACT_CHECK_API_KEY")
    }
    response = requests.get(FACT_CHECK_URL, params=params)
    return response.json().get("claims", [])

# ============== MAIN WORKFLOW ==============
def process_user_input(user_text):
    pipeline = load_pipeline()  # Load pipeline here
    prediction = pipeline.predict([user_text])[0]
    claims = extract_claims_with_openai(user_text)
    return {
        "prediction": "Fake" if prediction == 1 else "Real",
        "claims": claims,
        "fact_checks": [{"claim": c, "fact_check": google_fact_check(c)} for c in claims]
    }
