

##########################
###   Configure path   ###
##########################

DATA_ROOT_PATH=/data/zeng/webface_256x256
TRAIN_LIST=webface_256x256_list/train.txt
VAL_LIST=webface_256x256_list/val.txt
#DATA_ROOT_PATH=ck_data
#TRAIN_LIST=ck_data/train.txt
#VAL_LIST=ck_data/val.txt
SAVE_PATH=/data/zeng/pytorch_model/

############################
###   Image parameters   ###
############################

IMAGE_SIZE=256
CROP_SIZE=256


############################
###   Model parameters   ###
############################

NUM_CLASSES=10572
MODEL=sphere20
BATCH_SIZE=256
LR=0.1
EPOCH=18


CUDA_VISIBLE_DEVICES=2,3 python pytorch_train.py \
  --root_path=${DATA_ROOT_PATH} \
  --train_list=${TRAIN_LIST} \
  --val_list=${VAL_LIST} \
  --save_path=${SAVE_PATH} \
  --model=$MODEL  \
  --num_classes=$NUM_CLASSES \
  --img_size=$IMAGE_SIZE \
  --crop_size=$CROP_SIZE \
  --batch-size=$BATCH_SIZE \
  --lr=$LR \
  --epochs=$EPOCH

