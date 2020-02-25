#!/usr/bin/env bash

readonly CURRENT_DIR=$(dirname $(realpath $0))
readonly DATA_PATH_BASE=$(realpath ${CURRENT_DIR}/../data)
readonly DATA_PATH=${DATA_PATH_BASE}/voc
if [ ! -d ${DATA_PATH} ]; then
    mkdir -p ${DATA_PATH}
fi

DOWNLOAD_FILES='
http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
'

for download_file in ${DOWNLOAD_FILES}; do
    file_name=$(basename $download_file)
    if [ ! -f $DATA_PATH/$file_name ]; then
        wget ${download_file} -P $DATA_PATH
        extension="${file_name##*.}"
        if [[ ${extension} == "zip" ]]; then
            unzip $DATA_PATH/${file_name} -d ${DATA_PATH}
        fi

        if [[ ${extension} == "tar" ]]; then
            tar xvf $DATA_PATH/${file_name} -C ${DATA_PATH}
        fi
    fi
done
