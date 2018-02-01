#!/usr/bin/env sh

#task=WuNet_mobile
#task=depthwise_test
#task=deepid
#task=wfface
task=inception-resnet_v1
#task=senet50
#task=wenet_force
#task=sparse_Wen

#MODEL_HOME=/root/caffe/my_dirtywork/model
CAFFE_HOME=/home/zeng/wf-caffe

${CAFFE_HOME}/build/tools/caffe train \
  --solver=model/${task}/solver.prototxt -gpu 2,3
  #--solver=${MODEL_HOME}/${task}/solver.prototxt

