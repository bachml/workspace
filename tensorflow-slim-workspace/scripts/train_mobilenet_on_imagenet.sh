#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a lenet model on the Imagenet training set.
# 2. Evaluates the model on the Imagenet validation set.
#
# Usage:
# ./scripts/train_wenet_on_imagenet.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/GFace8_model+

# Where the dataset is saved to.
DATASET_DIR=/data/zeng/tf_webface_256x256

# Run training.
CUDA_VISIBLE_DEVICES=0,1,2,3 python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=imagenet \
  --dataset_split_name=wenet \
  --dataset_dir=${DATASET_DIR} \
  --model_name=mobilenet_v2_025 \
  --preprocessing_name=face_preprocessing \
  --width_multiplier=1.0 \
  --max_number_of_steps=28000 \
  --batch_size=64 \
  --save_interval_secs=189 \
  --save_summaries_secs=300 \
  --log_every_n_steps=100 \
  --optimizer=sgd \
  --momentum=0.9 \
  --weight_decay=0.0005 \
  --learning_rate_decay_type=step \
  --num_clones=4 \
  --learning_rate=0.1
  #--rmsprop_decay=0.9
  #--num_epochs_per_decay=30.0 \
  #--learning_rate_decay_factor=0.1 \
  #--opt_epsilon=1.0\

## Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=imagenet \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#/  --model_name=wenet
