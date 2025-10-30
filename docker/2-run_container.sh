#!/bin/bash

from_tag="20.12-py3-asr"
image="nvcr.io/nvidia/pytorch"
# image="asr/pytorch"

container_tag=${from_tag}
echo "Using image ${image}:${container_tag}."

this_time="$(date '+%Y%m%dT%H%M')"
if [ -z "$( which nvidia-docker )" ]; then
    cmd0="docker run --gpus all"
else
    cmd0="NV_GPU='${docker_gpu}' nvidia-docker run "
fi
container_name="cloud-qa-${docker_gpu//,/_}_${this_time}"

cmd0="${cmd0} -it --rm --net=host --ipc=host \
    -w /dialogue_api \
    -v /data/asr/hoangtm/cloud-callcenter-qa:/cloud-callcenter-qa"

this_env=""

if [ ! -z "${docker_env}" ]; then
    docker_env=$(echo ${docker_env} | tr "," "\n")
    for i in ${docker_env[@]}
    do
        this_env="-e $i ${this_env}" 
    done
fi

if [ ! -z "${HTTP_PROXY}" ]; then
    this_env="${this_env} -e 'HTTP_PROXY=${HTTP_PROXY}'"
fi

if [ ! -z "${http_proxy}" ]; then
    this_env="${this_env} -e 'http_proxy=${http_proxy}'"
fi

cmd="${cmd0} -it --rm ${this_env} --name ${container_name} ${vols} ${image}:${container_tag}"

trap ctrl_c INT

function ctrl_c() {
    echo "** Kill docker container ${container_name}"
    docker rm -f ${container_name}
}

echo "Executing application in Docker"
echo ${cmd}
eval ${cmd}

echo "`basename $0` done."
