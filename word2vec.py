import os
import argparse
from tqdm import tqdm

from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

from preprocess import Preprocessor
from utils import write_txt


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
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--year",
        type=int_list,
        required=True,
    )
    parser.add_argument("--month", type=int_list_2d, required=True)
    parser.add_argument(
        "--negative", type=int, help="how many noise words should be drawn", default=5
    )
    parser.add_argument(
        "--min_count", type=int, help="how many noise words should be drawn", default=5
    )
    parser.add_argument(
        "--epochs", type=int, help="Number of iterations over the corpus", default=5
    )
    parser.add_argument(
        "--vector_size", type=int, help="Dimensionality of the word vectors", default=30
    )
    parser.add_argument("--window", type=int, help="context window size", default=5)

    parser.add_argument("--save_preprocessed_corpus", action="store_true")
    parser.add_argument("--rewrite_saved_corpus", action="store_true")

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


def preprocess(args, year, month):
    if os.path.isfile(f"{args.data_dir}/{year}_merged/{month}_preprocessed_w2v.txt"):
        print(f"\nPreprocessed corpus for {year}/{month} already exsits!")
        if not args.rewrite_saved_corpus:
            return

    preprocessor = Preprocessor(remove_stopwords=False)
    corpus = preprocessor.preprocess(
        data_path=f"{args.data_dir}/{year}_merged/{month}_merged.csv"
    )  # list of sentences

    if args.save_preprocessed_corpus:
        write_txt(f"{args.data_dir}/{year}_merged/{month}_preprocessed_w2v.txt", corpus)


def gen_vectors(args, year, month):
    model = Word2Vec(
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        negative=args.negative,
    )
    print(f"\nModel initialized - {year}/{month}")

    corpus = LineSentence(f"{args.data_dir}/{year}_merged/{month}_preprocessed_w2v.txt")
    model.build_vocab(corpus)
    print("Vocabulary initialized")

    model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Training finished")

    model.wv.save_word2vec_format(f"{args.output_dir}/vector/{year}_{month}.w2v")
    print("Vectors saved")

    model.save(f"{args.output_dir}/model/{year}_{month}.model")
    print("Model saved")


def calc_last(year, month):
    if month == 1:
        last_month = 12
        last_year = year - 1
    else:
        last_month = month - 1
        last_year = year
    return last_year, last_month


def gen_vectors_preinit(args, year, month):
    corpus_t = LineSentence(f"{args.data_dir}/{year}_merged/{month}_preprocessed_w2v.txt")
    print(f"Dataset Loaded - {year}/{month}")

    last_year, last_month = calc_last(year, month)
    model = Word2Vec.load(f"{args.output_dir}/model/{last_year}_{last_month}.model")
    print("Vectors pre-initialized")

    model.build_vocab(corpus_t, update=True)
    print("Vocabulary initialized")

    model.train(corpus_t, total_examples=model.corpus_count, epochs=model.epochs)
    print("Train finished")

    model.wv.save_word2vec_format(f"{args.output_dir}/vector/{year}_{month}.w2v")
    print("Vectors saved")

    model.save(f"{args.output_dir}/model/{year}_{month}.model")
    print("Model saved")


if __name__ == "__main__":
    args = get_args()
    os.makedirs(f"{args.output_dir}/vector", exist_ok=True)
    os.makedirs(f"{args.output_dir}/model", exist_ok=True)

    year_0 = args.year[0]
    month_0 = args.month[0].pop(0)

    preprocess(args, year_0, month_0)
    gen_vectors(args, year_0, month_0)

    for i in tqdm(range(0, len(args.year))):
        for j in tqdm(range(0, len(args.month[i]))):
            preprocess(args, args.year[i], args.month[i][j])
            gen_vectors_preinit(args, args.year[i], args.month[i][j])
