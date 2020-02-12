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
    sys.exit(-1)


def main():
    data_path = os.path.join(_CURRENT_DIR, "../saved_models")
    os.system("mkdir -p {}".format(data_path))
    file_id = "1_-FQFU1i79WySBehqdUXAdbI_-RSvFqb"
    destination = os.path.join(data_path, "darknet53.conv.74")
    if not os.path.isfile(destination):
        download_file_from_google_drive(file_id, destination)


if __name__ == "__main__":
    main()
