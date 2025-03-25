from flask import Flask, request, jsonify
from joblib import load

# Load the trained model from the saved file
model = load("sentiment_model.pkl")

# Create a Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    # Predict sentiment using the trained model
    prediction = model.predict([text])[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    # Vectorize and predict
    vect_text = vectorizer.transform([text])
    prediction = model.predict(vect_text)[0]
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
