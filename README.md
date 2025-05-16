# 🧠 Flask Sentiment Analysis App

This is a Flask-based web application that performs sentiment analysis on user input using a trained machine learning model.

---

## ✨ Features

- Takes text input and predicts its sentiment (**Positive** or **Negative**)
- Uses a trained **TF-IDF + Logistic Regression** model for sentiment analysis
- Built with **Flask** for the backend and basic HTML for frontend
- Lightweight and easy to run locally

---

## 🗂️ Files in This Repository

| File                   | Description                                |
|------------------------|--------------------------------------------|
| `app.py`               | Main Flask web application                 |
| `sentiment_analysis.py`| Contains the sentiment prediction logic    |
| `sentiment_data.csv`   | Dataset used for training/testing          |
| `sentiment_model.pkl`  | Trained ML model (Logistic Regression)     |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer for text preprocessing   |
| `templates/index.html` | HTML template for user input               |

---

## ⚙️ How to Run the Application

### Step 1: Clone the repository
```bash
git clone https://github.com/subhashree212/flask-sentiment-app.git
cd flask-sentiment-app
