#!/usr/bin/env sh
TASK=Wen_ECCV
set -e
CAFFE_HOME=/home/zeng/dp-caffe

${CAFFE_HOME}/build/tools/caffe train \
    --solver=model/${TASK}/solver.prototxt -gpu all \
    --snapshot=buffer_/${TASK}_iter_21701.solverstate
    #--snapshot=trained_model/${TASK}_iter_32035.solverstate
