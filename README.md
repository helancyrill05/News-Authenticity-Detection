
Project Name: News Authenticity Detection

Description
This project implements an end-to-end machine learning pipeline to classify news articles as real or fake using 9 textual parameters. Five algorithms (SVM, Logistic Regression, Gradient Boosting, Decision Tree, Naive Bayes) were compared with SVM achieving the highest accuracy of 97% (95% precision, 96% recall). The final SVM model is deployed in a Streamlit demo for real-time predictions. The solution integrates OpenAI API for automated claim extraction and Google Fact Check API for external verification, using Python tools like Scikit-learn, Pandas, and NumPy.

Features
End-to-end ML pipeline: Data preprocessing, feature engineering, model training, and evaluation
Algorithm comparison: SVM, Logistic Regression, Gradient Boosting, Decision Tree, Naive Bayes
Final Model:SVM 
Basic Demo: Streamlit app for real-time classification
External verification: OpenAI API + Google Fact Check API integration
Comprehensive metrics: Accuracy, precision, recall, F1-score, AUC scores, and confusion matrices
Visualization: ROC curves, confusion matrices, feature importance plots

Dataset
Parameters Used: 9 textual features (e.g., TF-IDF vectors, sentiment scores, readability metrics, etc.)

Project Structure and File Descriptions
1.fake news detection proj.ipynb: This Jupyter Notebook contains the complete workflow for the project, including exploratory data analysis (EDA), data preprocessing, feature engineering, model training and evaluation, and visualization of results such as confusion matrices and ROC curves.
2.app.py:A Streamlit-based interactive web application that uses the trained SVM model to provide real-time fake news predictions. It also integrates OpenAI API for claim extraction and Google Fact Check API for external verification.
3.predict_usernews.py:A Python script designed to take user input or news text and generate fake news predictions using the trained model. This script can be used independently or integrated into other applications.
4.requirements.txt:Lists all the Python dependencies and packages required to run the project, including libraries for data processing, machine learning, API access, and the Streamlit app.

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
