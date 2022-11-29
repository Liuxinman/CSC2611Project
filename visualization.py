import csv
import pickle
import math
import argparse
import numpy as np
from tqdm import tqdm
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.lines as mlines

from sklearn.manifold import TSNE
from adjustText import adjust_text
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

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

    def str_list(x):
        return x.split(",")

    parser = argparse.ArgumentParser()

    parser.add_argument("--keyword_fpath", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tf_fpath", type=str, required=True)
    parser.add_argument("--merged_emb_fpath", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--year",
        type=int_list,
        required=True,
    )
    parser.add_argument(
        "--ts_year",
        type=int_list,
        required=True,
    )
    parser.add_argument("--month", type=int_list_2d, required=True)
    parser.add_argument("--ts_month", type=int_list_2d, required=True)
    parser.add_argument("--plot_topn", type=int, default=2)
    parser.add_argument("--tsne_topn", type=int, default=30)
    parser.add_argument("--vs_keyword", type=str_list, default=[])
    parser.add_argument("--semantic_change_keyword", type=str_list, default=[])
    parser.add_argument("--tsne_keyword", type=str_list, default=[])
    args = parser.parse_args()

    if print_args:
        _print_args(args)
    return args


class TsnePlotter:
    def __init__(self, model_path, output_dir, keys, year, month, plot_topn=5, tsne_topn=30):
        self.keys = keys
        self.model_path = model_path
        self.plot_topn = plot_topn
        self.tsne_topn = tsne_topn
        self.year = year
        self.month = month
        self.output_dir = output_dir

    def create_tsne_plot(self):
        if len(self.keys) == 0:
            return
        emb_2d_all = []  # #intervals * #keywords * emb dim
        word_cluster_all = []  # #intervals * #keywords
        for i in tqdm(range(len(self.year))):
            for j in tqdm(range(len(self.month[i]))):
                emb_clusters, word_clusters = self.gen_emb_clusters(
                    f"{self.model_path}/{self.year[i]}_{self.month[i][j]}.w2v",
                    topn=self.tsne_topn,
                )
                emb_2d = self.tsne_2d(emb_clusters)
                emb_2d_all.append(emb_2d)
                word_cluster_all.append(word_clusters)

        labels = self._gen_labels()
        for i, keyword in enumerate(self.keys):
            emb_2d_key = [
                emb_2d_all[j][i, :, :] for j in range(len(emb_2d_all))
            ]  # #intervals * emb_dim
            word_cluster_key = [word_cluster_all[j][i] for j in range(len(word_cluster_all))]
            word_cluster_new, emb_2d_new = self.create_diachronic_emb_2d(
                emb_2d_key, word_cluster_key, topn=self.plot_topn
            )
            self.tsne_plot_diachronic(emb_2d_new, word_cluster_new, labels, keyword)

    def gen_emb_clusters(self, model_path, topn=30):
        model_gn = KeyedVectors.load_word2vec_format(
            model_path,
            binary=False,
        )

        embedding_clusters = []
        word_clusters = []
        for word in self.keys:
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

    def tsne_2d(self, embedding_clusters):
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

    def _gen_labels(self):
        labels = []
        for i in range(len(self.year)):
            for j in range(len(self.month[i])):
                labels.append(f"{self.year[i]}/{self.month[i][j]}")
        return labels

    def create_diachronic_emb_2d(self, emb_2d_all, word_clusters_all, topn=10):
        word_cluster_new = [[] for _ in range(len(emb_2d_all))]
        emb_2d_new = np.zeros((len(emb_2d_all), topn + 1, 2))
        for i in range(len(emb_2d_all)):  # only consider the first keyword now
            keyword = word_clusters_all[i][0]
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

    def tsne_plot_diachronic(self, emb_2d_clusters, word_clusters, labels, keyword, a=0.8):
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
            label_ts.append(
                plt.text(
                    np.mean(x),
                    np.mean(y),
                    label,
                    fontsize=35,
                    color="royalblue",
                    fontweight="bold",
                    alpha=0.25,
                )
            )
            label_loc.append([np.mean(x), np.mean(y)])

        adjust_text(
            ts, x=emb_all[:, 0], y=emb_all[:, 1], arrowprops=dict(arrowstyle="-", color="k", lw=0.5)
        )
        label_loc = np.array(label_loc)
        adjust_text(label_ts, x=label_loc[:, 0], y=label_loc[:, 1])

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

        plt.title(f"Semantic Change - {keyword.capitalize()}", fontsize=25, fontweight="bold")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/{keyword}.png")


class TimeSeriesPlotter:
    def __init__(
        self,
        tf_fpath,
        merged_emb_fpath,
        model_path,
        output_dir,
        year,
        month,
        batch_size=10000,
    ):
        self.output_dir = output_dir
        self.month = month
        self.year = year
        self.model_path = model_path

        df = pd.read_csv(tf_fpath)

        self.tf_idf_vocab = df["vocab"].to_list()
        wf = []
        tf_idf = []
        for i in range(len(year)):
            for j in range(len(month[i])):
                wf.append(df[f"tf_{year[i]}_{month[i][j]}"].to_list())
                tf_idf.append(df[f"tf_idf_{year[i]}_{month[i][j]}"].to_list())
        self.wf = np.array(wf).T
        self.tf_idf = np.array(tf_idf).T

        # retrieve word2vec embedding
        t = 0
        for i in range(len(year)):
            t += len(month[i])

        with open(merged_emb_fpath, "rb") as f:
            w2v_year, w2v_month, self.w2v_vocab, w2v_all = pickle.load(f)

        idxs = []
        month_len = [len(m) for m in w2v_month]
        for i in range(len(year)):
            for j in range(len(month[i])):
                idx = w2v_month[w2v_year.index(year[i])].index(month[i][j])
                idxs.append(idx + sum(month_len[: w2v_year.index(year[i])]))
        self.w2v = w2v_all[:, idxs, :]

        _, self.num_intervals = self.wf.shape

        self.ts = {}
        self.make_tf_idf_ts()
        print("wf and tf_idf time series is done!")
        self.make_w2v_ts(batch_size=batch_size)
        print("w2v time series is done!")

        self.calc_degree_of_change()
        print("degree of semantic change is done!")

    def make_tf_idf_ts(self):
        wf_ts = []
        tf_idf_ts = []
        for i, j in zip(range(self.num_intervals - 1), range(1, self.num_intervals)):
            wf_i, wf_j = self.wf[:, i], self.wf[:, j]
            tf_idf_i, tf_idf_j = self.tf_idf[:, i], self.tf_idf[:, j]
            wf_ts.append(wf_j - wf_i)
            tf_idf_ts.append(tf_idf_j - tf_idf_i)
        wf_ts = np.array(wf_ts).T
        tf_idf_ts = np.array(tf_idf_ts).T
        self.ts["wf"] = dict(zip(self.tf_idf_vocab, wf_ts))
        self.ts["tf_idf"] = dict(zip(self.tf_idf_vocab, tf_idf_ts))

    def make_w2v_ts(self, batch_size=10000):
        # use batch training (faster and doable)
        w2v_ts = np.zeros((self.w2v.shape[0], self.w2v.shape[1] - 1))
        start = 0
        for b in tqdm(range(1, math.ceil(self.w2v.shape[0] / batch_size) + 1)):
            end = min(b * batch_size, self.w2v.shape[0])
            for i, j in zip(range(self.num_intervals - 1), range(1, self.num_intervals)):
                w2v_i, w2v_j = self.w2v[start:end, i, :], self.w2v[start:end, j, :]
                w2v_ts[start:end, i] = np.diagonal(1 - cosine_similarity(w2v_i, w2v_j))
            start = end
        self.ts["w2v"] = dict(zip(self.w2v_vocab, w2v_ts))

    def make_tf_idf_vs_w2v_plot(self, key, ts1="tf_idf", ts2="w2v"):
        x_labels = []
        for m in self.month:
            x_labels += m
        x_labels = x_labels[1:]

        fig, ax = plt.subplots()
        ax.plot(range(1, self.num_intervals), self.ts[ts1][key], color="darkorange", marker="^")
        ax.set_xlabel("Month", fontsize=14)
        ax.set_xticks(range(1, self.num_intervals), x_labels)
        ax.set_ylabel("Delta TF", color="black", fontsize=12, fontweight="bold")
        legend1 = mlines.Line2D(
            [], [], color="darkorange", marker="^", markersize=10, label="Delta TF"
        )

        ax2 = ax.twinx()
        ax2.plot(range(1, self.num_intervals), self.ts[ts2][key], color="darkgreen", marker="p")
        ax2.set_ylabel("Delta W2V", color="black", fontsize=12, fontweight="bold")
        legend2 = mlines.Line2D(
            [], [], color="darkgreen", marker="p", markersize=10, label="Delta W2V"
        )
        ax.legend(handles=[legend1, legend2])
        ax.set_title("")

        fig.savefig(f"{self.output_dir}/tfidf_vs_w2v_{key}.png", bbox_inches="tight", dpi=300)

    def make_semantic_change_plot(self, key):
        def _find_most_sim_words(y, m):
            # find the most similar words
            model = KeyedVectors.load_word2vec_format(
                f"{self.model_path}/{y}_{m}.w2v",
                binary=False,
            )
            sims = [w for w, _ in model.most_similar(key, topn=3)]
            return sims

        sims = _find_most_sim_words(self.year[0], self.month[0][0]) + _find_most_sim_words(
            self.year[-1], self.month[-1][-1]
        )

        def _calc_cos_dist(emb1, emb2):
            emb1 = np.expand_dims(emb1, axis=0)
            emb2 = np.expand_dims(emb2, axis=0)
            return 1 - cosine_similarity(emb1, emb2)[0][0]

        cos_dist = np.zeros((len(sims), self.num_intervals))
        t = 0
        for i in range(len(self.year)):
            for j in range(len(self.month[i])):
                model = KeyedVectors.load_word2vec_format(
                    f"{self.model_path}/{self.year[i]}_{self.month[i][j]}.w2v",
                    binary=False,
                )
                emb_key = model[key]
                for w in range(len(sims)):
                    if sims[w] in model.key_to_index:
                        emb_w = model[sims[w]]
                        cos_dist[w, t] = _calc_cos_dist(emb_w, emb_key)
                    else:
                        cos_dist[w, t] = 1.0
                t += 1
        color = ["navy", "blue", "cornflowerblue", "orangered", "darkorange", "gold"]
        marker = ["p", ".", "o", "^", "x", "D"]
        plt.figure("semantic change")
        for w in range(len(sims)):
            plt.plot(
                range(1, self.num_intervals + 1),
                cos_dist[w, :],
                color=color[w],
                marker=marker[w],
                label=sims[w],
            )
        plt.legend(bbox_to_anchor=(0.5, 1.0), loc="upper center", borderaxespad=0.0, ncols=2)
        x_labels = []
        for m in self.month:
            x_labels += m
        plt.xticks(range(1, self.num_intervals + 1), x_labels)
        plt.ylim((0.0, 1.5))
        plt.xlabel("Month", fontweight="bold")
        plt.ylabel("Cosine Distance", fontweight="bold")
        plt.title(f"{key.capitalize()} Semantic Change", fontweight="bold", fontsize=14)
        plt.savefig(f"{self.output_dir}/{key}_semantic_change.png", dpi=300)

    def calc_degree_of_change(self, batch_size=10000):
        valid_idx = []
        vocab = []
        word_removed = []
        for i in range(len(self.w2v)):
            zero_emb = False
            for j in range(len(self.w2v[0])):
                if sum(self.w2v[i][j]) == 0:
                    zero_emb = True
                    word_removed.append(self.w2v_vocab[i])
                    break
            if not zero_emb:
                valid_idx.append(i)
                vocab.append(self.w2v_vocab[i])
        emb = np.array(self.w2v)[valid_idx]

        emb_start = normalize(emb[:, 0, :])
        emb_end = normalize(emb[:, -1, :])

        degree = np.zeros((emb.shape[0]))
        start = 0
        for b in tqdm(range(1, math.ceil(emb.shape[0] / batch_size) + 1)):
            end = min(b * batch_size, emb.shape[0])
            emb_0, emb_1 = emb_start[start:end, :], emb_end[start:end, :]
            degree[start:end] = np.diagonal(1 - cosine_similarity(emb_0, emb_1))
            start = end
        self.degree = sorted(list(zip(vocab, degree)), key=lambda x: x[1], reverse=True)

    def get_most_least_changing_words(self, start=0, end=100):
        print(f"************** The {start}-{end} most changing word **************")
        for i in range(start, end):
            print(self.degree[i])
        print(f"************** The {start}-{end} least changing word **************")
        for i in range(-start, -end - 1, -1):
            print(self.degree[i])

    def get_degree_rank(self, keys):
        rank = [d[0] for d in self.degree]
        print(f"Words in total: {len(rank)}")
        for key in keys:
            print(f"The degree of semantic change of {key} ranked {rank.index(key) + 1}")

    def make_degree_histogram(self):
        plt.figure("histogram")
        plt.hist(
            [d[1] for d in self.degree],
            30,
            density=True,
            stacked=True,
            facecolor="navy",
            alpha=0.75,
        )
        plt.xlim([-0.05, 1.05])
        plt.xlabel("Degree of Semantic Change")
        plt.title("Histogram of Degree of Semantic Change")
        plt.grid(True)
        plt.savefig(f"{self.output_dir}/semantic_change_histogram.png", dpi=300)


if __name__ == "__main__":
    args = get_args()

    tsne_plotter = TsnePlotter(
        model_path=args.model_path,
        output_dir=args.output_dir,
        keys=args.tsne_keyword,
        year=args.year,
        month=args.month,
        plot_topn=args.plot_topn,
        tsne_topn=args.tsne_topn,
    )
    tsne_plotter.create_tsne_plot()

    time_series_plotter = TimeSeriesPlotter(
        tf_fpath=args.tf_fpath,
        merged_emb_fpath=args.merged_emb_fpath,
        model_path=args.model_path,
        output_dir=args.output_dir,
        year=args.ts_year,
        month=args.ts_month,
    )
    for key in args.vs_keyword:
        time_series_plotter.make_tf_idf_vs_w2v_plot(key=key)
    for key in args.semantic_change_keyword:
        time_series_plotter.make_semantic_change_plot(key=key)

    time_series_plotter.get_most_least_changing_words(start=400, end=500)
    time_series_plotter.make_degree_histogram()
    time_series_plotter.get_degree_rank(args.tsne_keyword)
