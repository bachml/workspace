#!/usr/bin/env sh

#task=WuNet_mobile
task=Wen_ECCV
#task=wenet_force
#task=sparse_Wen

#MODEL_HOME=/root/caffe/my_dirtywork/model
CAFFE_HOME=/home/zeng/caffe_wyd

${CAFFE_HOME}/build/tools/caffe train \
  --solver=model/${task}/solver.prototxt -gpu all
  #--solver=${MODEL_HOME}/${task}/solver.prototxt

