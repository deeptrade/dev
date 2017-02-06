#!/bin/bash

if [ -z "$1" ]
then
    echo "Usage: run.sh directory, where directory is the directory to be mapped to /dev in the container"
    exit 1
fi

nvidia-docker run --name tensorflow -d -p 8888:8888 -p 6006:6006 -v ${1}:/code tensorflow/tensorflow:latest-gpu
