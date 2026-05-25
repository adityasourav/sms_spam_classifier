import nltk
import string

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()


def transform_text(text):

    text = str(text).lower()

    words = nltk.word_tokenize(text)

    y=[]

    for word in words:
        if word.isalnum():
            y.append(word)

    words = y[:]
    y.clear()

    for word in words:
        if word not in stopwords.words("english") and word not in string.punctuation:
            y.append(ps.stem(word))

    return " ".join(y)