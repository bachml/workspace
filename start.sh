#!/usr/bin/env sh

#task=WuNet_mobile
task=LightenedCNN
#task=wenet_force
#task=sparse_Wen

#MODEL_HOME=/root/caffe/my_dirtywork/model
CAFFE_HOME=/home/zeng/sfm-caffe

${CAFFE_HOME}/build/tools/caffe train \
  --solver=model/${task}/solver.prototxt -gpu all
  #--solver=${MODEL_HOME}/${task}/solver.prototxt

