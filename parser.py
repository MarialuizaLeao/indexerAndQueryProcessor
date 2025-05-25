from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk import download
import re

class Parser:
    def __init__(self):
        download('stopwords')
        download('punkt')
        download('punkt_tab')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()

    def tokenize_with_hyphen(self, text):
        tokens = re.findall(r'\b\w+(?:-\w+)*\b', text.lower())
        expanded = []
        for token in tokens:
            expanded.append(token)
            if '-' in token:
                expanded.extend(token.split('-'))
        return expanded

    def parse_text(self, text):
        tokens = word_tokenize(text.lower())
        filtered = [self.stemmer.stem(w) for w in tokens if w.isalnum() and w not in self.stop_words]
        for token in tokens:
            if '-' in token:
                filtered.append(self.stemmer.stem(token))
                expended = self.tokenize_with_hyphen(token)
                for word in expended:
                    if word.isalnum() and word not in self.stop_words:
                        filtered.append(self.stemmer.stem(word))
            elif token.isalnum() and token not in self.stop_words:
                filtered.append(self.stemmer.stem(token))
        return filtered
