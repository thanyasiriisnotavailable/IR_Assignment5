import re
from nltk import word_tokenize


def preProcess(s):
        s = s.lower()
        s = re.sub(r'[^A-Za-z]', ' ', s)
        s = re.sub(r'\s+', ' ', s)
        s = word_tokenize(s)
        return ' '.join(s)