#!/usr/bin/env sh

#task=WuNet_mobile
#task=depthwise_test
task=zengNet
#task=wenet_force
#task=sparse_Wen

#MODEL_HOME=/root/caffe/my_dirtywork/model
CAFFE_HOME=/home/zeng/myCaffe

${CAFFE_HOME}/build/tools/caffe train \
  --solver=model/${task}/solver.prototxt -gpu all
  #--solver=${MODEL_HOME}/${task}/solver.prototxt

