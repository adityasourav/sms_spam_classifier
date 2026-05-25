import pandas as pd
import pickle
import nltk
import string
import ssl

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Fix SSL issue (Mac)
ssl._create_default_https_context = ssl._create_unverified_context

# Download resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")

ps = PorterStemmer()


def transform_text(text):

    text = str(text).lower()

    words = nltk.word_tokenize(text)

    y = []

    for word in words:
        if word.isalnum():
            y.append(word)

    words = y[:]
    y.clear()

    for word in words:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(ps.stem(word))

    return " ".join(y)


print("Loading dataset...")

# IMPORTANT FIX
df = pd.read_csv(
    "../data/spam.csv",
    encoding="utf-16",
    engine="python"
)

print("\nColumns before cleaning:")
print(df.columns)


# Keep only first two columns
df = df.iloc[:, :2]

# Rename safely
df.columns = ["target", "text"]

# Remove null rows
df.dropna(inplace=True)

print("\nDataset after cleaning:")
print(df.head())


# Encode labels
encoder = LabelEncoder()

df["target"] = encoder.fit_transform(df["target"])


print("\nPreprocessing text...")

df["transformed_text"] = df["text"].apply(transform_text)


# TF-IDF
tfidf = TfidfVectorizer(max_features=3000)

X = tfidf.fit_transform(
    df["transformed_text"]
).toarray()

y = df["target"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=2
)


print("\nTraining model...")

model = MultinomialNB()

model.fit(X_train, y_train)

pred = model.predict(X_test)

accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)


# Save files
pickle.dump(
    model,
    open("../models/model.pkl", "wb")
)

pickle.dump(
    tfidf,
    open("../models/vectorizer.pkl", "wb")
)

print("\nModel saved successfully")