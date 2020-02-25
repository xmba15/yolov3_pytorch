#!/usr/bin/env bash

readonly WEIGHT_PATH="https://pjreddie.com/media/files/darknet53.conv.74"
readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_PATH=$(realpath ${CURRENT_DIR}/../saved_models)

function validate_url {
    wget --spider $1 &> /dev/null;
}

if ! validate_url $WEIGHT_PATH; then
    echo "Invalid url to download darknet weight";
    exit;
fi

echo "start downloading darknet weights"
if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
fi

if [ ! -f ${DATA_PATH}/darknet53.conv.74 ]; then
    wget -c ${WEIGHT_PATH} -P ${DATA_PATH}
fi
