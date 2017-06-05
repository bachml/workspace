#!/usr/bin/env sh

task=LightenedCNN_lmdb

CAFFE_HOME=/home/zeng/caffe_wyd

${CAFFE_HOME}/build/tools/caffe train \
  --solver=model/${task}/solver.prototxt \
  --weights trained_model/LightenedCNN_C.caffemodel

