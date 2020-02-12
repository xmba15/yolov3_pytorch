#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
try:
    sys.path.append(os.path.join(_CURRENT_DIR, ".."))
    from config import Config
except Exception as e:
    print(e)
    exit(1)


def main(args):
    import numpy as np
    from matplotlib import pyplot as plt

    total_config = Config()
    if not args.dataset or args.dataset not in total_config.DATASETS.keys():
        raise Exception("specify one of the datasets to use in {}".format(list(total_config.DATASETS.keys())))

    DatasetClass = total_config.DATASETS[args.dataset]

    dataset = DatasetClass(data_path=total_config.DATA_PATH)
    size_list = dataset.size_distribution()
    hist, bins = np.histogram(size_list, bins=args.num_bins)

    print("min size {}".format(min(size_list)))
    print("max size {}".format(max(size_list)))
    print(hist, bins)
    plt.hist(size_list, bins=args.num_bins)
    plt.title("size/distance histogram")
    # plt.show()
    plt.savefig("size_distance_hist.png")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="name of the dataset to use")
    parser.add_argument("--num_bins", type=int, default=18, help="number of bins in histogram")
    parsed_args = parser.parse_args()

    main(parsed_args)
