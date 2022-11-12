import os
import math
import argparse
from tqdm import tqdm

from gensim.corpora import Dictionary

from preprocess import Preprocessor
from utils import read_csv, write_csv, read_parquet, write_parquet


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

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument(
        "--year",
        type=int_list,
        required=True,
    )
    parser.add_argument("--month", type=int_list_2d, required=True)
    parser.add_argument("--min_count", type=int, default=5)

    args = parser.parse_args()

    if print_args:
        _print_args(args)
    return args


def _print_args(args):
    """Print arguments."""
    print("------------------------ arguments ------------------------", flush=True)
    str_list = []
    for arg in vars(args):
        dots = "." * (48 - len(arg))
        str_list.append("  {} {} {}".format(arg, dots, getattr(args, arg)))
    for arg in sorted(str_list, key=lambda x: x.lower()):
        print(arg, flush=True)
    print("-------------------- end of arguments ---------------------", flush=True)


def gen_vocab(data_dir, year, month, min_freq=5):
    if os.path.isfile(f"{data_dir}/vocab.csv"):
        vocab = read_csv(f"{data_dir}/vocab.csv", key="vocab")
        return vocab

    corpus = []
    preprocessor = Preprocessor(remove_stopwords=True)
    for i in tqdm(range(len(year))):
        for j in tqdm(range(len(month[i]))):
            corpus_m = preprocessor.preprocess(
                data_path=f"{data_dir}/{year[i]}_merged/{month[i][j]}_merged.csv"
            )  # list of sentences
            write_parquet(
                path=f"{data_dir}/{year[i]}_merged/{month[i][j]}_preprocessed.parquet",
                data_dct={"corpus": corpus_m},
            )
            corpus += corpus_m

    dct = Dictionary(corpus)
    token2id = dct.token2id
    cfs = dct.cfs
    vocab = [key for key in token2id if cfs[token2id[key]] >= min_freq]
    print(f"Vocabulary Size: {len(vocab)}")

    write_csv(path=f"{data_dir}/vocab.csv", data_dct={"vocab": vocab})

    return vocab


if __name__ == "__main__":
    args = get_args()

    # generate vocabulary
    vocab = gen_vocab(args.data_dir, args.year, args.month, args.min_count)

    stat_dct = {"vocab": vocab}
    # for each w in V, calc monthly word freq and tf-idf
    for i in tqdm(range(len(args.year))):
        for j in tqdm(range(len(args.month[i]))):
            corpus_ij = read_parquet(
                path=f"{args.data_dir}/{args.year[i]}_merged/{args.month[i][j]}_preprocessed.parquet",
                key="corpus",
            )
            dct_ij = Dictionary(corpus_ij)
            term_freq_ij = []
            tf_idf_ij = []
            term_freq_sum = sum(list(dct_ij.cfs.values()))
            for w in vocab:
                w_id = dct_ij.token2id.get(w)
                if w_id:
                    term_freq_ij.append(dct_ij.cfs[w_id] / term_freq_sum)
                    tf_idf_ij.append(
                        math.log(dct_ij.cfs[w_id]) * math.log(dct_ij.num_docs / dct_ij.dfs[w_id])
                    )
                else:
                    term_freq_ij.append(0)
                    tf_idf_ij.append(0)
            stat_dct[f"tf_{args.year[i]}_{args.month[i][j]}"] = term_freq_ij
            stat_dct[f"tf_idf_{args.year[i]}_{args.month[i][j]}"] = tf_idf_ij

    write_csv(path=f"{args.data_dir}/vocab_stat.csv", data_dct=stat_dct)
