#!/usr/bin/env sh
TASK=depthwise_test
set -e
CAFFE_HOME=/home/zeng/dp-caffe

${CAFFE_HOME}/build/tools/caffe train \
    --solver=model/${TASK}/solver.prototxt -gpu all \
    --snapshot=buffer_/${TASK}_iter_58.solverstate
    #--snapshot=trained_model/${TASK}_iter_32035.solverstate
