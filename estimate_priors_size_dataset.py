#!/usr/bin/env python
# -*- coding: utf-8 -*-


def main(args):
    import numpy as np
    from config import Config
    from utils import estimate_priors_sizes

    total_config = Config()
    if not args.dataset or args.dataset not in total_config.DATASETS.keys():
        raise Exception("specify one of the datasets to use in {}".format(list(total_config.DATASETS.keys())))

    DatasetClass = total_config.DATASETS[args.dataset]

    dataset = DatasetClass(data_path=total_config.DATA_PATH)
    clusters = estimate_priors_sizes(dataset, args.k)
    clusters = [list(elem) for elem in sorted(clusters, key=lambda x: x[0], reverse=False)]
    print("Anchors:")
    print(clusters)

    anchor_masks = list(np.arange(args.k)[::-1].reshape(3, -1))
    anchor_masks = [list(elem[::-1]) for elem in anchor_masks]
    print("Anchor masks:")
    print(anchor_masks)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=9, help="number of clusters")
    parser.add_argument("--dataset", type=str, required=True, help="name of the dataset to use")
    parsed_args = parser.parse_args()

    main(parsed_args)
