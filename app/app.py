
from flask import Flask, render_template, request, redirect, url_for, flash
import joblib
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download stopwords if not already
nltk.download('stopwords')

app = Flask(__name__)
app.secret_key = 'super_secret_key'

# Load the model and vectorizer
try:
    model = joblib.load('../model/spam_classifier.pkl')  # Adjust path if needed
    vectorizer = joblib.load('../model/tfidf_vectorizer.pkl')
except Exception as e:
    print(f"Model loading failed: {e}")
    model = None
    vectorizer = None

# Text preprocessing function
def clean_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return ' '.join(tokens)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    if request.method == "POST":
        if 'refresh' in request.form:
            return redirect(url_for('index'))

        message = request.form["message"]

        if not model or not vectorizer:
            prediction = "⚠️ Model not loaded"
        else:
            try:
                cleaned = clean_text(message)
                vect = vectorizer.transform([cleaned])
                pred = model.predict(vect)[0]

                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(vect)[0]
                    if len(proba) == 2:
                        spam_proba = proba[1] * 100
                        if pred == 1:
                            prediction = f"🔴 Spam Detected ❌ ({spam_proba:.2f}%)"
                            flash("⚠️ Spam Detected!", "danger")
                        else:
                            prediction = f"✅ Ham ({100 - spam_proba:.2f}%)"
                            flash("✅ This message is safe.", "success")
                    else:

                        prediction = "⚠️ Model output not valid: Proba length mismatch"
                else:
                    prediction = "✅ Ham" if pred == 0 else "🔴 Spam"
            except Exception as e:
                prediction = f"Prediction Error ❗ {e}"
                print("Prediction Error:", e)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)
