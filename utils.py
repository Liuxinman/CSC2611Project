import pandas as pd


def read_csv(path, key="text"):
    df = pd.read_csv(path, lineterminator="\n")
    text = df[key].to_list()
    return text


def write_csv(path, data_dct):
    df = pd.DataFrame(data_dct)
    df.to_csv(path, index=False)


def read_parquet(path, key="corpus"):
    df = pd.read_parquet(path)
    text = df[key].to_list()
    return text


def write_parquet(path, data_dct):
    df = pd.DataFrame(data_dct)
    df.to_parquet(path, index=False)
