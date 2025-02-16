from nltk import word_tokenize
import re

def preProcess(s):
    s = s.lower()
    s = re.sub(r'[^A-Za-z\s]', '', s)  # Remove non-alphabetic characters, but keep spaces
    s = re.sub(r'\s+', ' ', s).strip()  # Remove extra spaces
    tokens = word_tokenize(s)  # Tokenize words properly
    return ' '.join(tokens)

def preprocess_stopwords(stop_words):
    return [' '.join(re.findall(r'[A-Za-z]+', word)) for word in stop_words]