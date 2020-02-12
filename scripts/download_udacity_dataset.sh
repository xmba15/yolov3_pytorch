#!/usr/bin/env bash

# https://github.com/BeSlower/Udacity_object_dataset
readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_PATH_BASE=$(realpath ${CURRENT_DIR}/../data)
readonly DATA_PATH=${DATA_PATH_BASE}/udacity

echo "start downloading udacity dataset"
if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
fi

if [ ! -f ${DATA_PATH}/object-dataset.tar.gz ]; then
    wget -c https://s3.amazonaws.com/udacity-sdc/annotations/object-dataset.tar.gz -P ${DATA_PATH}
fi

tar -xvf ${DATA_PATH}/object-dataset.tar.gz -C ${DATA_PATH}
