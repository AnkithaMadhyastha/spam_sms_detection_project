# Spam SMS Detection
A simple spam classifier using Naive Bayes and Flask web interface.
# 📩 SMS Spam Detection Web App

A simple web-based application built using Flask and Machine Learning to detect whether a given SMS message is Spam or Ham (Not Spam).

---

## 🚀 Features

- ✅ Predicts whether an SMS is spam or not
- ⚠️ Flash alert for spam messages
- 🧼 Clean text preprocessing using NLTK
- 🔍 Real-time text prediction using a trained ML model
- 🔄 Refresh/reset button
- 🖥️ Simple, clean Bootstrap UI
- 🔒 Lightweight and secure Flask backend

---

## 🧠 How It Works

1. **Input**: User enters an SMS message on the web interface.
2. **Preprocessing**: Message is lowercased, cleaned of punctuation, stopwords are removed, and stemming is applied.
3. **Vectorization**: Cleaned text is transformed using a trained **TF-IDF vectorizer**.
4. **Prediction**: The **Multinomial Naive Bayes** model predicts if the message is spam or ham.
5. **Output**: The prediction is shown with a flash message and result card.

---


## 📁 Folder Structure   
spam_sms_detection_project/ ├── app/ │ ├── app.py # Flask application │ ├── templates/ │ │ └── index.html # Frontend ├── model/ │ ├── spam_classifier.pkl # Trained ML model │ └── tfidf_vectorizer.pkl # TF-IDF Vectorizer ├── spam.csv # Dataset (optional) ├── requirements.txt └── README.md

---

## 📦 Installation

 Clone the repository:

git clone https://github.com/AnkithaMadhyastha/spam-detector-app.git
cd spam-detector-app


---

## 📊 Dataset Used

The system uses the **SMS Spam Collection Dataset**, a labeled dataset of SMS messages marked as either `spam` or `ham`. It is widely used for NLP tasks and is included in the project as `spam.csv`.

**Columns:**
- `label` – spam or ham
- `message` – text of the SMS

---

## 💡 Sample Inputs

### ✅ Ham Message Example:
Hey! Just checking in. Are we still on for lunch today?

### ❌ Spam Message Example:
Congratulations! You’ve won a free vacation to Bahamas! Call now to claim.

---

## 🎯 Output Format

When a message is entered, the app returns:

- ✅ **Ham** – if the message is safe
- ❌ **Spam** – if it's a spam message

A flash message and label appear on the web page accordingly.

---
## 🔬 Model Training Info

The model was trained using:

- **Algorithm**: `Multinomial Naive Bayes`
- **Vectorizer**: `TfidfVectorizer`
- **Text Cleanup**:
  - Lowercasing
  - Removing punctuation
  - Removing stopwords
  - Applying stemming (`PorterStemmer`)

---
## 📦 License
Generate the `spam_classifier.pkl` and `tfidf_vectorizer.pkl` model training script for you too.
This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute it with credit to the author.