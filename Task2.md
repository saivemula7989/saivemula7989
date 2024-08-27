# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load the movie review dataset (replace with your own data)
data = pd.read_csv('movie_reviews.csv')

# Define features (X) and target variable (y)
X = data['review']  # feature: movie review text
y = data['sentiment']  # target variable: sentiment (0 = negative, 1 = positive)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a TF-IDF vectorizer to convert text into numerical features
vectorizer = TfidfVectorizer(stop_words='english')

# Fit the vectorizer to the training data and transform both datasets
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Create a Naive Bayes classifier to predict sentiment
clf = MultinomialNB()

# Train the classifier on the training data
clf.fit(X_train_tfidf, y_train)

# Make predictions on the testing data
y_pred = clf.predict(X_test_tfidf)

# Evaluate the classifier's performance
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred)
