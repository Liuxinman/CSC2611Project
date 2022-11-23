import csv
import argparse
import numpy as np
from tqdm import tqdm

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.manifold import TSNE
from adjustText import adjust_text
from gensim.models import KeyedVectors

from utils import _print_args


def get_args(print_args=True):
    def int_list(x):
        # 10,11,12
        # [10, 11, 12]
        return list(map(int, x.split(",")))

    def int_list_2d(x):
        # 10,11,12;1,2,3,4,5
        # [[10, 11, 12], [1, 2, 3, 4, 5]]
        return [list(map(int, r.split(","))) for r in x.split(";")]

    parser = argparse.ArgumentParser()

    parser.add_argument("--keyword_fpath", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--year",
        type=int_list,
        required=True,
    )
    parser.add_argument("--month", type=int_list_2d, required=True)
    parser.add_argument("--plot_topn", type=int, default=2)
    parser.add_argument("--tsne_topn", type=int, default=30)
    args = parser.parse_args()

    if print_args:
        _print_args(args)
    return args


def gen_emb_clusters(model_path, keys, topn=30):
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
        embeddings.append(model_gn.get_vector(word, norm=True))
        for similar_word, _ in model_gn.most_similar(word, topn=topn):
            words.append(similar_word)
            embeddings.append(model_gn.get_vector(similar_word, norm=True))
        embedding_clusters.append(embeddings)
        word_clusters.append(words)
    return np.array(embedding_clusters), word_clusters


def tsne_2d(embedding_clusters):
    n, m, k = embedding_clusters.shape
    tsne_model_en_2d = TSNE(
        perplexity=10,
        n_components=2,
        init="random",
        n_iter=5000,
        learning_rate=150,
        random_state=32,
        n_jobs=-1,
    )
    embeddings_en_2d = np.array(
        tsne_model_en_2d.fit_transform(embedding_clusters.reshape(n * m, k))
    ).reshape(n, m, 2)
    return embeddings_en_2d


def gen_labels(args):
    labels = []
    for i in range(len(args.year)):
        for j in range(len(args.month[i])):
            labels.append(f"{args.year[i]}/{args.month[i][j]}")
    return labels


def create_diachronic_emb_2d(emb_2d_all, word_clusters_all, topn=10):
    word_cluster_new = [[] for _ in range(len(emb_2d_all))]
    emb_2d_new = np.zeros((len(emb_2d_all), topn + 1, 2))
    for i in range(len(emb_2d_all)):  # only consider the first keyword now
        keyword = word_clusters_all[i][0]
        labels.append(f"{keyword}{i}")
        kv = KeyedVectors(vector_size=2)
        kv.add_vectors(word_clusters_all[i], emb_2d_all[i])
        word_cluster_new[i].append(keyword)
        emb_2d_new[i, 0, :] = kv.get_vector(keyword, norm=True)
        j = 1
        for w, _ in kv.most_similar(keyword, topn=topn):
            word_cluster_new[i].append(w)
            emb_2d_new[i, j, :] = kv.get_vector(w, norm=True)
            j += 1
    return word_cluster_new, emb_2d_new


def tsne_plot_diachronic(args, emb_2d_clusters, word_clusters, labels, keyword, a=0.8):
    figsize = (
        (9.5, 6) if (matplotlib.get_backend() == "nbAgg") else (20, 12)
    )  # interactive plot should be smaller
    plt.figure(figsize=(figsize))
    colors = cm.gist_rainbow(np.linspace(0, 1, len(labels)))
    ts = []
    emb_all = np.concatenate(emb_2d_clusters, axis=0)
    label_ts = []
    label_loc = []
    for label, embeddings, words, color in zip(labels, emb_2d_clusters, word_clusters, colors):
        x = embeddings[:, 0]
        y = embeddings[:, 1]
        plt.scatter(x, y, c=[color], alpha=a, label=label, s=20)
        for i in range(len(words)):
            ts.append(plt.text(x[i], y[i], words[i], fontsize=13))
        label_ts.append(plt.text(np.mean(x), np.mean(y), label, fontsize=35, color="royalblue", fontweight="bold", alpha=0.25))
        label_loc.append([np.mean(x), np.mean(y)])
    
    adjust_text(
        ts, x=emb_all[:, 0], y=emb_all[:, 1], arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
    )
    label_loc = np.array(label_loc)
    adjust_text(
        label_ts, x=label_loc[:, 0], y=label_loc[:, 1])

    # draw arrows between consecutive time intervals
    for i in range(len(emb_2d_clusters) - 1):
        plt.arrow(
            x=emb_2d_clusters[i, 0, 0],
            y=emb_2d_clusters[i, 0, 1],
            dx=emb_2d_clusters[i + 1, 0, 0] - emb_2d_clusters[i, 0, 0],
            dy=emb_2d_clusters[i + 1, 0, 1] - emb_2d_clusters[i, 0, 1],
            color="k",
            width=0.002,
            head_width=0.03,
            head_length=0.06,
            length_includes_head=True,
            alpha=0.3,
        )

    # plt.legend()
    plt.title(f"Semantic Change - {keyword.capitalize()}", fontsize=25, fontweight="bold")
    plt.grid(True)
    plt.savefig(f"{args.output_dir}/{keyword}.png")


if __name__ == "__main__":
    args = get_args()

    keys = []
    with open(args.keyword_fpath, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            keys.append(row[0])

    emb_2d_all = []  # #intervals * #keywords * emb dim
    word_cluster_all = []  # #intervals * #keywords
    for i in tqdm(range(len(args.year))):
        for j in tqdm(range(len(args.month[i]))):
            emb_clusters, word_clusters = gen_emb_clusters(
                f"{args.model_path}/{args.year[i]}_{args.month[i][j]}.w2v",
                keys,
                topn=args.tsne_topn,
            )
            emb_2d = tsne_2d(emb_clusters)
            emb_2d_all.append(emb_2d)
            word_cluster_all.append(word_clusters)

    labels = gen_labels(args)
    for i, keyword in enumerate(keys):
        emb_2d_key = [
            emb_2d_all[j][i, :, :] for j in range(len(emb_2d_all))
        ]  # #intervals * emb_dim
        word_cluster_key = [word_cluster_all[j][i] for j in range(len(word_cluster_all))]
        word_cluster_new, emb_2d_new = create_diachronic_emb_2d(
            emb_2d_key, word_cluster_key, topn=args.plot_topn
        )
        tsne_plot_diachronic(args, emb_2d_new, word_cluster_new, labels, keyword)
