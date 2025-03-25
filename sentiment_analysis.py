import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from joblib import dump

# Load the dataset from CSV
df = pd.read_csv("sentiment_data.csv")

# Prepare the training data (X = input text, y = labels)
X, y = df["text"], df["label"]

# Create a pipeline with TF-IDF vectorizer and Naive Bayes classifier
model = make_pipeline(TfidfVectorizer(), MultinomialNB())

# Train the model on the dataset
model.fit(X, y)

# Save the trained model to a file
dump(model, "sentiment_model.pkl")

print("Model trained and saved as sentiment_model.pkl")
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib
import nltk
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("sentiment_data.csv")
X = df['text']
y = df['label']

# Load stopwords
stop_words = stopwords.words('english')

# Convert text to numerical features using TF-IDF
vectorizer = TfidfVectorizer(stop_words=stop_words, max_features=5000)
X_vectorized = vectorizer.fit_transform(X)

# Split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model and vectorizer
joblib.dump(model, "sentiment_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
