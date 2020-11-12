import random
from pathlib import Path
from argparse import ArgumentParser

def split_data(input_path: str, output_dir: str):
    random.seed(46)
    data = []
    with open(input_path) as f:
        data.extend(f.readlines())

    random.shuffle(data)

    data_length = len(data)

    with (Path(output_dir) / "train.tsv").open("w") as f:
        f.writelines(data[:int(data_length / 10 * 8)])

    with (Path(output_dir) / "dev.tsv").open("w") as f:
        f.writelines(data[int(data_length / 10 * 8):int(data_length / 10 * 9)])

    with (Path(output_dir) / "test.tsv").open("w") as f:
        f.writelines(data[int(data_length / 10 * 9):])

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_dir", type=str)

    args = parser.parse_args()
    split_data(args.input_path, args.output_dir)
