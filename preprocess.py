import re
import string
import pandas as pd
from tqdm import tqdm

import nltk

nltk.download("stopwords")
nltk.download("punkt")

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

from utils import read_csv


class Preprocessor:
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords

    def _contain_punc(self, word):
        punctuation = [p for p in string.punctuation]
        for p in punctuation:
            if p in word:
                return True
        return False

    def _contain_alpha(self, word):
        return re.search("[a-zA-Z]+", word)

    def _remove_emoji(self, text):
        regrex_pattern = re.compile(
            pattern="["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE,
        )
        return regrex_pattern.sub(r"", text)

    def preprocess(self, data_path):
        corpus = read_csv(data_path)

        # lower-case, stem, remove punctuation, remove emoji
        tokenized_sent = []
        ps = PorterStemmer()
        stop_words = stopwords.words("english")
        for s in tqdm(corpus):
            sent = []
            for w in word_tokenize(s):
                w = self._remove_emoji(w)
                if (
                    len(w) > 0
                    and self._contain_alpha(w)
                    and not self._contain_punc(w)
                    and ((not self.remove_stopwords) or (w not in stop_words))
                ):
                    sent.append(ps.stem(w.lower()))
            tokenized_sent.append(sent)

        return tokenized_sent
