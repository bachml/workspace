#!/usr/bin/env sh
TASK=sphereface
set -e
CAFFE_HOME=/home/zeng/myCaffe

${CAFFE_HOME}/build/tools/caffe train \
    --solver=model/${TASK}/solver.prototxt -gpu all \
    --snapshot=buffer_/${TASK}_iter_10000.solverstate
    #--snapshot=trained_model/${TASK}_iter_32035.solverstate
