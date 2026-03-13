# 🧠 Flask Sentiment Analysis App

A Flask-based REST API that predicts sentiment (Positive/Negative) from text input using a trained Machine Learning model (TF-IDF + Logistic Regression).

## ✨ Features
- Accepts text input and returns sentiment prediction
- Trained TF-IDF + Logistic Regression model
- Lightweight REST API built with Flask
- Easy to run locally

## 🗂️ Project Structure
| File | Description |
|------|-------------|
| `app.py` | Main Flask API |
| `sentiment_analysis.py` | Sentiment prediction logic |
| `sentiment_data.csv` | Training dataset |
| `sentiment_model.pkl` | Trained ML model |
| `tfidf_vectorizer.pkl` | TF-IDF vectorizer |

## ⚙️ How to Run

### Step 1: Clone the repository
```
git clone https://github.com/subhashree212/flask-sentiment-app.git
cd flask-sentiment-app
```

### Step 2: Install dependencies
```
pip install -r requirements.txt
```

### Step 3: Run the app
```
python app.py
```

### Step 4: Test the API
```
curl -X POST http://127.0.0.1:5000/predict \
-H "Content-Type: application/json" \
-d '{"text": "I love this app!"}'
```

### Response
```json
{
  "sentiment": "positive"
}
```

## 🛠️ Tech Stack
- Python
- Flask
- scikit-learn
- TF-IDF Vectorizer
- Logistic Regression
