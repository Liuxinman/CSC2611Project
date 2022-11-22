import numpy as np
import argparse
import csv
import numpy as np
import plotly.graph_objs as go
from sklearn.manifold import TSNE
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from gensim.models import KeyedVectors

from utils import _print_args


def gen_emb_clusters(model_path, keys):
    model_gn = KeyedVectors.load_word2vec_format(
        model_path,
        binary=False,
    )

    embedding_clusters = []
    word_clusters = []
    for word in keys:
        embeddings = []
        words = []
        words.append(word)
        embeddings.append(model_gn[word])
        for similar_word, _ in model_gn.most_similar(word, topn=30):
            words.append(similar_word)
            embeddings.append(model_gn[similar_word])
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    return np.array(embedding_clusters), word_clusters


def tsne_2d(embedding_clusters):
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(
        perplexity=10,
        n_components=2,
        init="pca",
        n_iter=5000,
        learning_rate=150,
        random_state=32,
        n_jobs=-1,
    )
    embeddings_en_2d = np.array(
        tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    ).reshape(n, m, 2)
    return embeddings_en_2d


def create_diachronic_emb_ed(emb_2d_all, word_clusters_all, topn=10):
    word_cluster_new = [[] for _ in range(len(emb_2d_all))]
    emb_2d_new = np.zeros((len(emb_2d_all), topn + 1, 2))
    labels = []
    for i in range(len(emb_2d_all)): # only consider the first keyword now
        keyword = word_clusters_all[i][0][0]
        labels.append(f"{keyword}{i}")
        kv = KeyedVectors(vector_size=2)
        kv.add_vectors(word_clusters_all[i][0], emb_2d_all[i][0, :, :])
        word_cluster_new[i].append(keyword)
        emb_2d_new[i, 0, :] = kv[keyword]
        j = 1
        for w, _ in kv.most_similar(keyword, topn=topn):
            print(w, j)
            word_cluster_new[i].append(w)
            emb_2d_new[i, j, :] = kv[w]
            j += 1
    print(word_cluster_new)
    print(emb_2d_new)
    return word_cluster_new, emb_2d_new, labels


keys = ["lockdown", "panic", "virtual", "quarantine"]
emb_2d_all = []
word_cluster_all = []
year = [2019, 2020]
month = [[10, 11, 12], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
for i in range(len(year)):
    for j in range(len(month[i])):
        emb_clusters, word_clusters = gen_emb_clusters(
            f"/Users/xinmanliu/Documents/CourseUT/2611/project/CSC2611Project/output/vector/{year[i]}_{month[i][j]}.w2v",
            keys,
        )
        emb_2d = tsne_2d(emb_clusters)
        emb_2d_all.append(emb_2d)
        word_cluster_all.append(word_clusters)
word_cluster_new, emb_2d_new, labels = create_diachronic_emb_ed(emb_2d_all, word_cluster_all, topn=2)


def tsne_plot_diachronic(emb_2d_clusters, word_clusters, labels, a=0.7, title="diachronic word embedding"):
    figsize = (
        (9.5, 6) if (matplotlib.get_backend() == "nbAgg") else (20, 12)
    )  # interactive plot should be smaller
    plt.figure(figsize=(figsize))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, emb_2d_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=[color], alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(
                word,
                alpha=0.5,
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    plt.show()


def tsne_plot_similar_words(title, labels, embedding_clusters, word_clusters, a, filename=None):
    figsize = (
        (9.5, 6) if (matplotlib.get_backend() == "nbAgg") else (20, 12)
    )  # interactive plot should be smaller
    plt.figure(figsize=(figsize))
    colors = cm.rainbow(np.linspace(0, 1, len(labels)))
    for label, embeddings, words, color in zip(labels, embedding_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=[color], alpha=a, label=label)
        for i, word in enumerate(words):
            plt.annotate(
                word,
                alpha=0.5,
                xy=(x[i], y[i]),
                xytext=(5, 2),
                textcoords="offset points",
                ha="right",
                va="bottom",
                size=8,
            )
    plt.legend(loc=4)
    plt.title(title)
    plt.grid(True)
    plt.show()


# tsne_plot_similar_words(
#     "Similar words from Google News",
#     keys,
#     embeddings_en_2d,
#     word_clusters,
#     0.7,
#     "similar_words.png",
# )

tsne_plot_diachronic(emb_2d_new, word_cluster_new, labels)
