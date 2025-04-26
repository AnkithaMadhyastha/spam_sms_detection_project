import os
import pandas as pd
import string
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Download NLTK stopwords if not already present
nltk.download('stopwords')

# 🔍 Load dataset
print("🔍 Loading dataset...")
df = pd.read_csv("data/spam_70ham_30spam.csv", encoding='latin1')

# Drop unnecessary columns and missing values
df.dropna(inplace=True)
df = df[['v1', 'v2']]  # Keep only relevant columns
print("Columns in dataset:", df.columns)

# Rename relevant columns
df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# 🧹 Clean and preprocess text
print("🧹 Cleaning and preprocessing text...")

def clean_text(text):
    text = text.lower()
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

df['cleaned_message'] = df['message'].apply(clean_text)

# 📊 Train/test split and vectorization
print("📊 Splitting and vectorizing...")
X = df['cleaned_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer()
X_train_vect = vectorizer.fit_transform(X_train)
X_test_vect = vectorizer.transform(X_test)

# 🚀 Train model
print("🚀 Training the Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train_vect, y_train)

# 📈 Evaluate model
print("\n📈 Model Evaluation:")
y_pred = model.predict(X_test_vect)
print(classification_report(y_test, y_pred))

# Additional metrics
print("\n🔄 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\n📊 Accuracy:", accuracy_score(y_test, y_pred))

# 📉 Check if both classes (0 and 1) are predicted before calculating ROC-AUC
if len(set(y_pred)) > 1:
    print("\n📉 ROC-AUC Score:", roc_auc_score(y_test, model.predict_proba(X_test_vect)[:, 1]))
else:
    print("\n📉 ROC-AUC Score is not calculated because only one class was predicted.")

# 💾 Save model and vectorizer
print("💾 Saving model and vectorizer...")

# ✅ Create model directory if it doesn't exist
os.makedirs("model", exist_ok=True)

joblib.dump(model, "model/spam_classifier.pkl")
joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")

print("✅ Model training complete and files saved successfully!")
