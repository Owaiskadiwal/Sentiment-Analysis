# Sentiment-Analysis
A real-time sentiment analysis web app that classifies user-submitted text using machine learning models like SVM, Random Forest, and Naive Bayes. Built with Python, scikit-learn, TF-IDF, and Flask, the system also compares model performance using accuracy, precision, recall, and F1-score.
# Sentiment Analysis System

A complete machine learning-based web application for real-time sentiment classification. This project allows users to input text and get instant sentiment predictions (Positive, Negative, or Neutral) and also compare different ML models based on key evaluation metrics.

---

## 🔍 Project Overview

This system is designed to classify text sentiments using popular machine learning models and compare their performance. It provides a user-friendly web interface using Flask for both prediction and performance comparison.

---

## 🚀 Features

- Real-time sentiment prediction via `/predict` route
- Multiple model comparison via `/compare` route
- Supports SVM, Random Forest, Naive Bayes, Logistic Regression, and more
- Evaluation using Accuracy, Precision, Recall, and F1-Score
- Visualizations: bar charts and confusion matrix heatmaps
- Modular design for easy updates and expansion

---

## 🧠 Machine Learning Models Used

- Support Vector Machine (SVM) ✅ (Best performing model)
- Random Forest
- Naive Bayes
- Gradient Boosting
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Decision Tree

---

## 📊 Results Summary (SVM)

| Metric         | Score |
|----------------|-------|
| Accuracy       | 97.17%   |
| Precision      | 0.97  |
| Recall         | 0.96  |
| F1-Score       | 0.98  |

SVM outperformed other models across all metrics, making it the primary choice for live prediction.

---

## 🛠️ System Components

- **DataLoader** – Loads and processes CSV data
- **TextVectorizer** – Converts text to TF-IDF feature vectors
- **SentimentClassifier** – Trains ML models and makes predictions
- **ModelComparator** – Compares all models with detailed metrics
- **Flask Web App** – For real-time input and visualization
