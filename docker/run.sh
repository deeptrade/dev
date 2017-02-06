#!/bin/bash

nvidia-docker run -it --rm -p 6006:6006 -v /home/ubuntu/dev:/src gcr.io/tensorflow/tensorflow:latest-gpu bash

