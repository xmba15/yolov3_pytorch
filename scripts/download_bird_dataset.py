#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys


_CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(_CURRENT_DIR, ".."))
try:
    from utils import download_file_from_google_drive
except Exception as e:
    print(e)
    exit(0)


def main():
    data_path = os.path.join(_CURRENT_DIR, "../data")
    file_id = "16IQjiGu-jl2oTqr5wsp9MmJxtQiuyIWq"
    destination = os.path.join(data_path, "bird_dataset.zip")
    if not os.path.isfile(destination) and not os.path.isdir(os.path.join(data_path, "bird_dataset")):
        download_file_from_google_drive(file_id, destination)
        os.system("cd {} && unzip bird_dataset.zip".format(data_path))


if __name__ == "__main__":
    main()
