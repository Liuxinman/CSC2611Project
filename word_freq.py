import string
import re
import os
import math
import numpy as np
import pandas as pd
import nltk

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from gensim.corpora import Dictionary
from tqdm import tqdm


def contain_punc(word):
    punctuation = [p for p in string.punctuation]
    for p in punctuation:
        if p in word:
            return True
    return False


def contain_alpha(word):
    return re.search("[a-zA-Z]+", word)


def remove_emoji(text):
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


def read_csv(path):
    df = pd.read_csv(path, lineterminator="\n")
    text = df["text"].to_list()
    return text


def preprocess(corpus, remove_stopwords=True):
    # lower-case, stem, remove punctuation, remove emoji
    tokenized_sent = []
    ps = PorterStemmer()
    stop_words = stopwords.words("english")
    for s in tqdm(corpus):
        sent = []
        for w in word_tokenize(s):
            w = remove_emoji(w)
            if (
                len(w) > 0
                and contain_alpha(w)
                and not contain_punc(w)
                and ((not remove_stopwords) or (w not in stop_words))
            ):
                sent.append(ps.stem(w.lower()))
        tokenized_sent.append(sent)

    return tokenized_sent


def gen_vocab(data_dir, year, month, min_freq=5):
    if os.path.isfile(f"{data_dir}/vocab.csv"):
        vocab_df = pd.read_csv(f"{data_dir}/vocab.csv")
        vocab = vocab_df["vocab"].to_list()
        return vocab
    corpus = []
    for i in tqdm(range(len(year))):
        for j in tqdm(range(len(month[i]))):
            corpus_m = read_csv(f"{data_dir}/{year[i]}_merged/{month[i][j]}_merged.csv")
            corpus_m = preprocess(corpus_m)  # list of sentences
            df = pd.DataFrame({"corpus": corpus_m})
            df.to_parquet(
                f"{data_dir}/{year[i]}_merged/{month[i][j]}_preprocessed.parquet", index=False
            )
            corpus += corpus_m

    dct = Dictionary(corpus)
    token2id = dct.token2id
    cfs = dct.cfs
    vocab = [key for key in token2id if cfs[token2id[key]] >= min_freq]
    print(f"Vocabulary Size: {len(vocab)}")

    df = pd.DataFrame({"vocab": vocab})
    df.to_csv(f"{data_dir}/vocab.csv", index=False)

    return vocab


if __name__ == "__main__":
    # generate vocabulary
    year = [2019, 2020]
    month = [[10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
    data_dir = "/Users/xinmanliu/Documents/CourseUT/2611/project/CSC2611Project/tweet_dataset"
    vocab = gen_vocab(data_dir, year, month)

    stat_dct = {"vocab": vocab}
    # for each w in V, calc monthly word freq and tf-idf
    for i in tqdm(range(len(year))):
        for j in tqdm(range(len(month[i]))):
            corpus_ij = pd.read_parquet(
                f"{data_dir}/{year[i]}_merged/{month[i][j]}_preprocessed.parquet"
            )
            corpus_ij = corpus_ij["corpus"].tolist()
            dct_ij = Dictionary(corpus_ij)
            term_freq_ij = []
            tf_idf_ij = []
            term_freq_sum = sum(list(dct_ij.cfs.values()))
            for w in vocab:
                w_id = dct_ij.token2id.get(w)
                if w_id:
                    term_freq_ij.append(dct_ij.cfs[w_id] / term_freq_sum)
                    tf_idf_ij.append(
                        math.log(dct_ij.cfs[w_id])
                        * math.log(dct_ij.num_docs / dct_ij.dfs[w_id])
                    )
                else:
                    term_freq_ij.append(0)
                    tf_idf_ij.append(0)
            stat_dct[f"tf_{year[i]}_{month[i][j]}"] = term_freq_ij
            stat_dct[f"tf_idf_{year[i]}_{month[i][j]}"] = tf_idf_ij
    stat_df = pd.DataFrame(stat_dct)
    stat_df.to_csv(f"{data_dir}/vocab_stat.csv")
