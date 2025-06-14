Fake News Detection

Overview
This project implements an end-to-end machine learning pipeline to classify news articles as real or fake using 9 textual parameters. Five algorithms (SVM, Logistic Regression, Gradient Boosting, Decision Tree, Naive Bayes) were compared with SVM achieving the highest accuracy of 97% (95% precision, 96% recall). The final SVM model is deployed in a Streamlit demo for real-time predictions. The solution integrates OpenAI API for automated claim extraction and Google Fact Check API for external verification, using Python tools like Scikit-learn, Pandas, and NumPy.

Features
End-to-end ML pipeline: Data preprocessing, feature engineering, model training, and evaluation

Algorithm comparison: SVM, Logistic Regression, Gradient Boosting, Decision Tree, Naive Bayes

Interactive demo: Streamlit app for real-time classification

External verification: OpenAI API + Google Fact Check API integration

Comprehensive metrics: Accuracy, precision, recall, F1-score, AUC scores, and confusion matrices

Visualization: ROC curves, confusion matrices, feature importance plots

Dataset

Size: 40,000 news articles

Parameters Used: 9 textual features (e.g., TF-IDF vectors, sentiment scores, readability metrics, etc.)

Technologies Used
Programming Language: Python 3.x

Data Handling & Analysis: Pandas, NumPy

Machine Learning & Model Evaluation: Scikit-learn, TF-IDF Vectorizer

Visualization: Matplotlib, Seaborn

NLP: NLTK

APIs: OpenAI API, Google Fact Check API

Deployment/Demo: Streamlit

Environment Management: python-dotenv

Others: joblib
