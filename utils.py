import os
import pandas as pd


def read_csv(path, key="text"):
    if os.path.isfile(path):
        df = pd.read_csv(path, lineterminator="\n")
        text = df[key].to_list()
        return text
    else:
        print("read_csv: invalid file path!")
        return None


def write_csv(path, data_dct):
    df = pd.DataFrame(data_dct)
    df.to_csv(path, index=False)


def read_parquet(path, key="corpus"):
    if os.path.isfile(path):
        df = pd.read_parquet(path)
        text = df[key].to_list()
        return text
    else:
        print("read_parquet: invalid file path!")
        return None


def write_parquet(path, data_dct):
    df = pd.DataFrame(data_dct)
    df.to_parquet(path, index=False)


def write_txt(path, data):
    with open(path, "w") as f:
        data = [" ".join(text) for text in data]
        f.write("\n".join(data))


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

