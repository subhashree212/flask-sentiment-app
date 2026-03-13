from flask import Flask, request, jsonify
import joblib

# Load the trained model and TF-IDF vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Create a Flask app
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)
    text = data.get("text", "")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Transform the input text using the same vectorizer used during training
    vect_text = vectorizer.transform([text])

    # Make prediction
    prediction = model.predict(vect_text)[0]

    # Return the prediction as JSON
    return jsonify({"sentiment": prediction})

if __name__ == "__main__":
    app.run(debug=True)
