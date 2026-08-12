import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


# Download required NLTK resources if missing
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)

try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)


ps = PorterStemmer()


def transform_text(text):
    text = str(text).lower()

    words = nltk.word_tokenize(text)

    filtered_words = []

    for word in words:
        if word.isalnum():
            filtered_words.append(word)

    processed_words = []

    for word in filtered_words:
        if word not in stopwords.words("english"):
            processed_words.append(ps.stem(word))

    return " ".join(processed_words)