#!/bin/bash
#
# This script performs the following operations:
# 1. Trains a MobileNet model on the Imagenet training set.
# 2. Evaluates the model on the Imagenet validation set.
#
# Usage:
# ./scripts/train_deepid_on_imagenet.sh

# Where the checkpoint and logs will be saved to.
TRAIN_DIR=/tmp/deepid-model

# Where the dataset is saved to.
DATASET_DIR=/media/zehao/WD/Dataset/processed/ImageNet2012/imagenet-data

# Run training.
python train_image_classifier.py \
  --train_dir=${TRAIN_DIR} \
  --dataset_name=imagenet \
  --dataset_split_name=train \
  --dataset_dir=${DATASET_DIR} \
  --model_name=deepid \
  --preprocessing_name=deepid \
  --width_multiplier=1.0 \
  --max_number_of_steps=28000 \
  --batch_size=64 \
  --save_interval_secs=240 \
  --save_summaries_secs=240 \
  --log_every_n_steps=100 \
  --optimizer=momentum \
  --opt_epsilon=1.0\
  --momentum=0.9 \
  --weight_decay=0.005 \
  --learning_rate_decay_type=step \
  --num_clones=2 \
  --learning_rate=0.1 \
  --learning_rate_decay_factor=0.1 \
  --rmsprop_decay=0.9 \
  --num_epochs_per_decay=30.0 \

## Run evaluation.
#python eval_image_classifier.py \
#  --checkpoint_path=${TRAIN_DIR} \
#  --eval_dir=${TRAIN_DIR} \
#  --dataset_name=imagenet \
#  --dataset_split_name=validation \
#  --dataset_dir=${DATASET_DIR} \
#  --model_name=deepid
