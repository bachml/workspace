#!/usr/bin/env sh

task=sphereface

CAFFE_HOME=/home/zeng/myCaffe

${CAFFE_HOME}/build/tools/caffe train \
  --solver=model/${task}/solver.prototxt \
  --weights=buffer_/${task}_iter_pretrained.caffemodel -gpu all

