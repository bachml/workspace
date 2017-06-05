#!/usr/bin/env sh
TASK=Wen_ECCV
set -e
CAFFE_HOME=/home/zeng/caffe_wyd

${CAFFE_HOME}/build/tools/caffe train \
    --solver=model/${TASK}/solver.prototxt -gpu all \
    --snapshot=_buffer/${TASK}_iter_28000.solverstate
    #--snapshot=trained_model/${TASK}_iter_32035.solverstate
